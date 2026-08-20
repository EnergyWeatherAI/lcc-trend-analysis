from dataclasses import dataclass
from pathlib import Path
import re
from typing import Generator
import warnings

import pandas as pd

from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.type_aliases import Timestamp

CERES_SYN1DEG_FILENAME_RE = re.compile(
    r"^CER_(?P<product>[^_]+)_(?P<platforms>[^_]+)_(?P<edition>Edition[^_]+)_"
    r"(?P<granule_id>\d+)\.(?P<date>\d{8})\.hdf$"
)


@dataclass(frozen=True)
class CeresSyn1DegFilenameInfo:
    product: str
    platforms: str
    edition: str
    granule_id: str
    time: Timestamp

@dataclass(frozen=True)
class CeresLevel0File:
    product: str
    platforms: str
    edition: str
    granule_id: str
    time: Timestamp
    file_path: Path


def _ceres_level0_path() -> Path:
    return get_data_paths().ceres

def parse_ceres_syn1deg_filename(file_path: Path) -> CeresSyn1DegFilenameInfo | None:
    """Parse a CERES SYN1deg level0 filename into its component fields.

    Example: CER_SYN1deg-Day_Terra-Aqua-NOAA20_Edition4B_416413.20260530.hdf
    """
    match = CERES_SYN1DEG_FILENAME_RE.match(file_path.name)
    if match is None:
        return None
    return CeresSyn1DegFilenameInfo(
        product=match.group("product"),
        platforms=match.group("platforms"),
        edition=match.group("edition"),
        granule_id=match.group("granule_id"),
        time=pd.to_datetime(match.group("date"), format="%Y%m%d"),
    )

def get_ceres_level0_files(
    product: str,
    year: int | None = None,
) -> Generator[CeresLevel0File, None, None]:
    """Discover CERES SYN1deg level0 files for a given product.

    Files are expected under level0/ceres/{product}/{YYYY}/*.hdf.
    """
    product_root = _ceres_level0_path() / product
    if year is not None:
        product_root = product_root / f"{year:04d}"

    if not product_root.exists():
        return

    for file_path in sorted(product_root.rglob(f"CER_{product}_*.hdf")):
        filename_info = parse_ceres_syn1deg_filename(file_path)
        if filename_info is None:
            warnings.warn(f"Could not parse CERES filename: {file_path.name}")
            continue

        yield CeresLevel0File(
            product=filename_info.product,
            platforms=filename_info.platforms,
            edition=filename_info.edition,
            granule_id=filename_info.granule_id,
            time=filename_info.time,
            file_path=file_path,
        )
