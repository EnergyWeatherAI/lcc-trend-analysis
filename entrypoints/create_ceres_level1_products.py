import hashlib
import logging
import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from lcc_trend_analysis.logging import get_logger, setup_logging
from lcc_trend_analysis.observations.ceres import (
	CERES_SYN1DEG_DAY_PRODUCT,
    extract_sites_from_ceres_grid,
	is_ceres_hourly_product,
    read_ceres_syn1deg_level0,
)
from lcc_trend_analysis.observations.utils import get_ground_sites_gdf
from lcc_trend_analysis.parallel_processing import parallel_map
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.type_aliases import GeoDataFrame

from .find_data.ceres import CeresLevel0File, get_ceres_level0_files

logger: logging.Logger = get_logger(__name__)

load_dotenv()
DATA_PATHS = get_data_paths()

TIMEOUT = 180.0
N_JOBS: int = int(os.environ.get("N_JOBS", multiprocessing.cpu_count()))
CERES_PRODUCT = "SYN1deg-1Hour"
TARGET_PATH_BASE: Path = DATA_PATHS.ceres_level1_product(CERES_PRODUCT)
YEAR_TO_PROCESS = None  # Set to an integer year to process only that year, or None to process all years

METADATA_COLUMNS = [
	"product",
	"platforms",
	"edition",
	"time",
	"rel_path",
	"source_data_path",
	"ground_site_signature",
	"matched_site_count",
]
METADATA_DEDUPE_COLUMNS = ["source_data_path"]


@dataclass(frozen=True)
class CeresLevel1FileMetadata:
	product: str
	platforms: str
	edition: str
	time: pd.Timestamp
	rel_path: Path
	source_data_path: str
	ground_site_signature: str
	matched_site_count: int


def _empty_metadata_dataframe() -> pd.DataFrame:
	return pd.DataFrame(columns=METADATA_COLUMNS)


def _metadata_to_dataframe(results: list[CeresLevel1FileMetadata]) -> pd.DataFrame:
	if not results:
		return _empty_metadata_dataframe()

	return pd.DataFrame(
		[
			{
				"product": result.product,
				"platforms": result.platforms,
				"edition": result.edition,
				"time": result.time,
				"rel_path": str(result.rel_path),
				"source_data_path": result.source_data_path,
				"ground_site_signature": result.ground_site_signature,
				"matched_site_count": result.matched_site_count,
			}
			for result in results
		]
	)


def _normalize_metadata_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
	if df is None or df.empty:
		return _empty_metadata_dataframe()

	df = df.copy()
	for column_name in METADATA_COLUMNS:
		if column_name not in df.columns:
			df[column_name] = pd.Series(dtype=object)

	df = df[METADATA_COLUMNS]
	df["product"] = df["product"].astype(str)
	df["platforms"] = df["platforms"].astype(str)
	df["edition"] = df["edition"].astype(str)
	df["rel_path"] = df["rel_path"].astype(str)
	df["source_data_path"] = df["source_data_path"].astype(str)
	df["ground_site_signature"] = df["ground_site_signature"].astype(str)
	df["matched_site_count"] = pd.to_numeric(
		df["matched_site_count"], errors="coerce"
	).fillna(0).astype(int)
	df["time"] = pd.to_datetime(df["time"])

	return (
		df.sort_values(by=METADATA_DEDUPE_COLUMNS, kind="mergesort")
		.drop_duplicates(subset=METADATA_DEDUPE_COLUMNS, keep="last")
		.reset_index(drop=True)
	)


def _ground_site_signature(ground_sites: GeoDataFrame) -> str:
	site_ids = sorted(str(site_id) for site_id in ground_sites.index.tolist())
	return hashlib.sha256("\n".join(site_ids).encode("utf-8")).hexdigest()


def _load_existing_metadata(product: str) -> pd.DataFrame:
	metadata_path = DATA_PATHS.ceres_level1_metadata(product)
	if not metadata_path.exists():
		return _empty_metadata_dataframe()
	return _normalize_metadata_dataframe(pd.read_parquet(metadata_path))


def _merge_metadata_frames(
	existing_metadata: pd.DataFrame,
	new_metadata: pd.DataFrame,
) -> pd.DataFrame:
	frames = [df for df in [existing_metadata, new_metadata] if df is not None and not df.empty]
	if not frames:
		return _empty_metadata_dataframe()
	merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
	return _normalize_metadata_dataframe(merged)


def _write_metadata_atomic(metadata_df: pd.DataFrame, metadata_path: Path) -> None:
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
	metadata_df.to_parquet(temporary_path, index=None)
	temporary_path.replace(metadata_path)


def _checkpointed_source_paths(
	metadata_df: pd.DataFrame,
	ground_site_signature: str,
) -> set[str]:
	if metadata_df.empty:
		return set()
	matching_rows = metadata_df[metadata_df["ground_site_signature"] == ground_site_signature]
	return {str(source_data_path) for source_data_path in matching_rows["source_data_path"]}


def _level1_output_path(file: CeresLevel0File) -> Path:
	return (
		TARGET_PATH_BASE
		/ file.time.strftime("%Y")
		/ file.time.strftime("%m")
		/ f"CER_{file.product}_{file.platforms}_{file.edition}.{file.time.strftime('%Y%m%d')}.level1.nc"
	)


def preprocess_and_store_ceres_file(
	file: CeresLevel0File,
	ground_sites: GeoDataFrame,
	ground_site_signature: str,
) -> CeresLevel1FileMetadata:
	logger.debug("Processing CERES file %s", file.file_path.name)

	ds_source = read_ceres_syn1deg_level0(
		file.file_path,
		product=CERES_PRODUCT,
		source_date=file.time,
	)
	level1_dataset = extract_sites_from_ceres_grid(ds_source, ground_sites)
	if "time" not in level1_dataset.dims:
		level1_dataset = level1_dataset.expand_dims(
			time=pd.DatetimeIndex([file.time.to_datetime64()])
		)
	ordered_dims = ["time", "site", "cloud_layer"] + [
		dim_name
		for dim_name in level1_dataset.dims
		if dim_name not in {"time", "site", "cloud_layer"}
	]
	level1_dataset = level1_dataset.transpose(*ordered_dims, missing_dims="ignore")

	target_path = _level1_output_path(file)
	target_path.parent.mkdir(parents=True, exist_ok=True)

	level1_dataset.attrs.update(
		{
			"ceres_product": CERES_PRODUCT,
			"product": file.product,
			"platforms": file.platforms,
			"edition": file.edition,
			"source_data_path": str(file.file_path),
			"time_representation": (
				"hourly_midpoint" if is_ceres_hourly_product(CERES_PRODUCT) else "daily_composite"
			),
			"title": (
				f"CERES {CERES_PRODUCT} level1 site extraction of cloud-layer fields "
				"at the nearest grid cell"
			),
		}
	)

	encoding = {
		str(var_name): {"dtype": "float32"} for var_name in level1_dataset.data_vars
	}
	level1_dataset.to_netcdf(
		target_path,
		mode="w",
		format="NETCDF4",
		engine="netcdf4",
		encoding=encoding,
	)

	return CeresLevel1FileMetadata(
		product=file.product,
		platforms=file.platforms,
		edition=file.edition,
		time=file.time,
		rel_path=target_path.relative_to(TARGET_PATH_BASE),
		source_data_path=str(file.file_path),
		ground_site_signature=ground_site_signature,
		matched_site_count=len(ground_sites),
	)


def run() -> None:
	ground_sites = get_ground_sites_gdf()
	ground_site_signature = _ground_site_signature(ground_sites)

	existing_metadata = _load_existing_metadata(CERES_PRODUCT)
	checkpointed_paths = _checkpointed_source_paths(existing_metadata, ground_site_signature)

	files_to_process = [
		file
		for file in get_ceres_level0_files(CERES_PRODUCT, year=YEAR_TO_PROCESS)
		if str(file.file_path) not in checkpointed_paths
	]
	logger.info("Found %d CERES %s files to process", len(files_to_process), CERES_PRODUCT)

	if not files_to_process:
		return

	results = parallel_map(
		partial(
			preprocess_and_store_ceres_file,
			ground_sites=ground_sites,
			ground_site_signature=ground_site_signature,
		),
		tasks=files_to_process,
		n_jobs=N_JOBS,
		timeout=TIMEOUT,
	)

	new_metadata = _metadata_to_dataframe(results)
	merged_metadata = _merge_metadata_frames(existing_metadata, new_metadata)
	_write_metadata_atomic(merged_metadata, DATA_PATHS.ceres_level1_metadata(CERES_PRODUCT))
	logger.info("Processed %d CERES files", len(results))


if __name__ == "__main__":
	setup_logging(logging.INFO)
	run()
