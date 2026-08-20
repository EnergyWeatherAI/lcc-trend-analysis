
from lcc_trend_analysis.type_aliases import Dataset


VARIABLE_METADATA = {
    "time": {
        "long_name": "Time (UTC)",
        "standard_name": "time",
    },
    "height": {
        "long_name": "Height above mean sea level",
        "standard_name": "height_above_mean_sea_level",
        "units": "m",
    },
    "range": {
        "long_name": "Range from instrument",
        "units": "m",
        "comment": "Distance from instrument to centre of each range bin.",
    },
    "latitude": {
        "long_name": "Latitude of site",
        "standard_name": "latitude",
        "units": "degrees_north",
    },
    "longitude": {
        "long_name": "Longitude of site",
        "standard_name": "longitude",
        "units": "degrees_east",
    },
    "altitude": {
        "long_name": "Altitude of site",
        "standard_name": "altitude",
        "units": "m",
    },
    "beta": {
        "long_name": "Attenuated backscatter coefficient",
        "units": "sr-1 m-1",
        "comment": "SNR-screened attenuated backscatter coefficient."
    },
    "beta_raw": {
        "long_name": "Attenuated backscatter coefficient",
        "units": "sr-1 m-1",
        "comment": "Non-screened attenuated backscatter coefficient."
    },
    "beta_smooth": {
        "long_name": "Attenuated backscatter coefficient",
        "units": "sr-1 m-1",
        "comment": "SNR-screened attenuated backscatter coefficient.\n"
        "Weak background smoothed using Gaussian 2D-kernel.",
    },
    "background_noise": {
        "long_name": "Background noise level",
        "units": "sr-1 m-1",
        "comment": "Background noise level estimated from the attenuated backscatter coefficient.",
    },
    "cloud_layer_mask": {
        "long_name": "Cloud layer mask",
        "units": "1",
        "comment": "Boolean cloud layer mask. Based on Tuononen et al. (2019) algorithm.",
    },
    "cloud_flag": {
        "long_name": "Cloud flag",
        "units": "1",
        "comment": "Boolean flag indicating presence of a cloud layer in the profile.",
    },
    "low_cloud_flag": {
        "long_name": "Low cloud flag",
        "units": "1",
        "comment": "Boolean flag indicating presence of a cloud layer in the profile with cloud base height < 2000 m a.g.l.",
    },
    "cloud_base_height_asl": {
        "long_name": "Cloud base height above mean sea level",
        "units": "m",
    },
    "cloud_base_height_agl": {
        "long_name": "Cloud base height above ground",
        "units": "m",
    },
    "fog_flag": {
        "long_name": "Fog detection flag",
        "units": "1",
        "comment": "Boolean flag indicating presence of fog based on integrated backscatter coefficient over first range gates.",
    },
    "cleaned_negative_profile_flag": {
        "long_name": "Cleaned negative profile flag",
        "units": "1",
        "comment": "Boolean flag indicating profiles that were cleaned based on consecutive negative backscatter values.",
    }
}

def set_variable_metadata(ds: Dataset) -> Dataset:
    for var_name, metadata in VARIABLE_METADATA.items():
        if var_name in ds.data_vars or var_name in ds.coords:
            for attr_name, attr_value in metadata.items():
                ds[var_name].attrs[attr_name] = attr_value
    return ds