from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _get_base_data_path() -> Path:
    base_data_path = os.environ.get("BASE_DATA_PATH")
    if not base_data_path:
        raise RuntimeError(
            "BASE_DATA_PATH must be set before accessing repository data paths."
        )
    return Path(base_data_path).expanduser().resolve()


@dataclass(frozen=True)
class DataPaths:
    base: Path

    @property
    def level0(self) -> Path:
        return self.base / "level0"

    @property
    def level1(self) -> Path:
        return self.base / "level1"

    @property
    def level2(self) -> Path:
        return self.base / "level2"

    @property
    def level3(self) -> Path:
        return self.base / "level3"

    @property
    def level4(self) -> Path:
        return self.base / "level4"

    @property
    def reference(self) -> Path:
        return self.base / "reference"

    @property
    def tmp(self) -> Path:
        return self.base / "tmp"

    def level2_clouds(self, dataset_name: str) -> Path:
        return self.level2 / dataset_name / "clouds"

    def level3_dataset(self, dataset_name: str) -> Path:
        return self.level3 / dataset_name

    def source(self, provider: str) -> Path:
        return self.level0 / provider

    @property
    def arm(self) -> Path:
        return self.source("arm")
    
    @property
    def ceres(self) -> Path:
        return self.source("ceres")

    @property
    def cloudnet(self) -> Path:
        return self.source("cloudnet")

    @property
    def cloudsat(self) -> Path:
        return self.source("cloudsat")

    @property
    def ceda(self) -> Path:
        return self.source("ceda")

    @property
    def ceda_badc(self) -> Path:
        return self.ceda / "badc"

    @property
    def chilbolton_ct75k(self) -> Path:
        return self.ceda_badc / "chilbolton" / "data" / "lidar-ct75k-all-years"

    @property
    def eprofile(self) -> Path:
        return self.ceda_badc / "e-profile"

    @property
    def era5(self) -> Path:
        return self.source("era5")

    @property
    def era5_n320(self) -> Path:
        return self.era5 / "N320"

    @property
    def fmi(self) -> Path:
        return self.source("fmi")

    @property
    def knmi(self) -> Path:
        return self.source("knmi")

    @property
    def meteo_swiss(self) -> Path:
        return self.source("meteoswiss")

    @property
    def meteo_swiss_lidar_chm15k(self) -> Path:
        return self.meteo_swiss / "LIDAR_CHM15k"

    @property
    def metno(self) -> Path:
        return self.source("metno")

    @property
    def niwa(self) -> Path:
        return self.source("niwa")

    @property
    def results(self) -> Path:
        return self.base / "results"

    @property
    def ground_sites(self) -> Path:
        return self.reference / "ground_sites.parquet"

    @property
    def spurious_data_ranges(self) -> Path:
        return self.reference / "spurious_data_ranges.json"
    
    @property
    def spurious_data_dates(self) -> Path:
        return self.reference / "spurious_data_dates.json"

    @property
    def calibration_breakpoints(self) -> Path:
        return self.reference / "calibration_breakpoints.json"

    @property
    def alc_level1b(self) -> Path:
        return self.level1 / "alc" / "l1b"

    @property
    def alc_level1b_source_metadata(self) -> Path:
        return self.alc_level1b / "alc_source_data_metadata.parquet"

    @property
    def alc_level1c(self) -> Path:
        return self.level1 / "alc" / "l1c"

    @property
    def alc_level1c_source_metadata(self) -> Path:
        return self.alc_level1c / "alc_source_data_metadata.parquet"

    @property
    def alc_level1c_preliminary(self) -> Path:
        return self.alc_level1c / "preliminary"

    @property
    def alc_level1c_preliminary_source_metadata(self) -> Path:
        return self.alc_level1c_preliminary / "alc_source_data_metadata.parquet"

    @property
    def alc_autocalibration(self) -> Path:
        return self.level1 / "alc" / "autocalibration"

    def alc_autocalibration_lidar_ratios(self, pass_index: int = 1) -> Path:
        if pass_index == 1:
            return self.alc_autocalibration / "lidar_ratios.parquet"
        if pass_index == 2:
            return self.alc_autocalibration / "lidar_ratios_pass2.parquet"
        raise ValueError(f"Unsupported autocalibration pass index: {pass_index}")

    def alc_autocalibration_factors(self, pass_index: int = 1) -> Path:
        if pass_index == 1:
            return self.alc_autocalibration / "autocalibration_factors.parquet"
        if pass_index == 2:
            return self.alc_autocalibration / "autocalibration_factors_pass2.parquet"
        raise ValueError(f"Unsupported autocalibration pass index: {pass_index}")

    @property
    def cloudsat_level1b(self) -> Path:
        return self.level1 / "cloudsat" / "l1b"

    @property
    def ceres_level1(self) -> Path:
        return self.level1 / "ceres" / "l1"

    def ceres_level1_product(self, product: str) -> Path:
        return self.ceres_level1 / product

    def ceres_level1_metadata(self, product: str) -> Path:
        return self.ceres_level1_product(product) / "ceres_level1_metadata.parquet"

    def ceres_level2_clouds(self, product: str) -> Path:
        return self.level2_clouds("ceres") / product

    def ceres_level2_clouds_raw(self, product: str) -> Path:
        return self.ceres_level2_clouds(product) / "ceres_level2_clouds_raw.nc"

    @property
    def alc_level2_clouds(self) -> Path:
        return self.level2_clouds("alc")

    def level2_clouds_raw(self, dataset_name: str) -> Path:
        return self.level2_clouds(dataset_name) / f"{dataset_name}_level2_clouds_raw.nc"

    @property
    def alc_level2_calibration_sensitivity(self) -> Path:
        return self.level2 / "alc" / "calibration_sensitivity"

    @property
    def alc_level2_calibration_sensitivity_dataset(self) -> Path:
        return (
            self.alc_level2_calibration_sensitivity
            / "alc_level2_calibration_sensitivity.nc"
        )

    @property
    def alc_level2_calibration_sensitivity_screened_catalog(self) -> Path:
        return (
            self.alc_level2_calibration_sensitivity
            / "screened_l1c_catalog.parquet"
        )

    @property
    def alc_level2_calibration_sensitivity_sample_catalog(self) -> Path:
        return (
            self.alc_level2_calibration_sensitivity
            / "synthetic_sample_catalog.parquet"
        )

    @property
    def alc_level2_calibration_sensitivity_scenarios(self) -> Path:
        return self.alc_level2_calibration_sensitivity / "drift_scenarios.parquet"

    @property
    def era5_level2_clouds(self) -> Path:
        return self.level2_clouds("era5")

    @property
    def era5_level2_clouds_hourly(self) -> Path:
        return self.era5_level2_clouds / "era5_level2_clouds_hourly.nc"

    def level3_clouds(self, dataset_name: str) -> Path:
        return self.level3_dataset(dataset_name) / "clouds"

    def ceres_level3_clouds(self, product: str) -> Path:
        return self.level3_clouds("ceres") / product

    def level3_level2_clouds_dataset(
        self,
        dataset_name: str,
        suffix: str = "",
        freq: str | None = None,
        anomaly: bool = False,
    ) -> Path:
        stem = f"{dataset_name}_level2_clouds"
        if freq:
            stem += f"_{freq}"
        if anomaly:
            stem += "_anomaly"
        return self.level3_clouds(dataset_name) / f"{stem}{suffix}.nc"

    def level3_cloud_cover_candidate_datasets(
        self,
        dataset_name: str,
        product_stem: str,
        suffix: str = "",
    ) -> list[Path]:
        base_path = self.level3_clouds(dataset_name)
        return [
            base_path / f"{dataset_name}_level2_clouds_{product_stem}{suffix}.nc",
            base_path / f"{dataset_name}_cloud_cover_{product_stem}{suffix}.nc",
        ]

    def level3_driver_anomaly(self, dataset_name: str, freq: str) -> Path:
        return self.level3_dataset(dataset_name) / f"{dataset_name}_{freq}_anomaly.nc"

    def cloudsat_level1b_product(self, product: str) -> Path:
        return self.cloudsat_level1b / product

    @property
    def cloudsat_overpass_records(self) -> Path:
        return self.cloudsat_level1b / "cloudsat_overpass_records.parquet"

    @property
    def level2_evaluation(self) -> Path:
        return self.level2 / "evaluation"

    def level2_evaluation_matchups(self, dataset_name: str) -> Path:
        return self.level2_evaluation / f"{dataset_name}_cloudsat_matchups.nc"

    @property
    def level3_evaluation(self) -> Path:
        return self.level3 / "evaluation"

    def level3_evaluation_histograms(self, dataset_name: str) -> Path:
        return self.level3_evaluation / f"{dataset_name}_cloudsat_joint_histograms.nc"

    @property
    def combined_cloud_profiles(self) -> Path:
        return self.level2_evaluation / "combined_cloud_profiles.nc"

    @property
    def level4_trends(self) -> Path:
        return self.level4 / "trends"

    def level4_trends_dataset(self, suffix: str = "") -> Path:
        return self.level4_trends / f"cloud_cover_trends{suffix}.nc"

    @property
    def level4_trends_slope_distributions(self) -> Path:
        return self.level4_trends / "slope_distributions"

    def level4_trends_slope_distribution_dataset(
        self, dataset_name: str, site: str, suffix: str = ""
    ) -> Path:
        return (
            self.level4_trends_slope_distributions
            / f"{dataset_name}_{site}_trend_slope_distribution{suffix}.parquet"
        )

    def level4_trends_seasonal_slope_distribution_dataset(
        self, dataset_name: str, site: str, season: str, suffix: str = ""
    ) -> Path:
        return (
            self.level4_trends_slope_distributions
            / f"{dataset_name}_{site}_{season}_trend_slope_distribution{suffix}.parquet"
        )

    @property
    def level4_responses(self) -> Path:
        return self.level4 / "responses"

    def level4_responses_dataset(self, suffix: str = "") -> Path:
        return self.level4_responses / f"cloud_cover_trend_and_responses{suffix}.nc"

@lru_cache(maxsize=1)
def get_data_paths() -> DataPaths:
    return DataPaths(base=_get_base_data_path())