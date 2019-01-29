
"""
Calculate additional properties (e.g., photometric and astrometric variability)
which we will use in our model, and create a subset file that only contains the
columns we need.
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table


input_path = "data/gaia-sources-for-npm.fits"
output_path = "data/gaia-sources-for-npm-colsubset.fits"

def create_subset_for_npm(path, hdu=1, additional_label_names=None, **kwargs):

    properties = dict()

    with fits.open(path) as image:

        sources = image[hdu].data

        for band in ("g", "bp", "rp"):
            # Absolute magnitudes, and photometric variability
            properties[f"absolute_{band}_mag"] = sources[f"phot_{band}_mean_mag"] \
                                               + 5 * np.log10(sources["parallax"]/100.0)

            properties[f"phot_{band}_variability"] = np.sqrt(sources["astrometric_n_good_obs_al"]) \
                                                   * sources[f"phot_{band}_mean_flux"] \
                                                   / sources[f"phot_{band}_mean_flux_error"]


        # Radial velocity scatter                                                   
        properties["rv_single_epoch_variance"] = sources["radial_velocity_error"]**2 \
                                               * sources["rv_nb_transits"] * np.pi/2.0
        properties["rv_single_epoch_scatter"] = properties["rv_single_epoch_variance"]**0.5
        
        # Astrometric unit weight error
        properties["astrometric_unit_weight_error"] = np.sqrt(
            sources["astrometric_chi2_al"]/(sources["astrometric_n_good_obs_al"] - 5))
        
        for label_name in additional_label_names:
            properties[label_name] = sources[label_name]
    
    return properties


data = create_subset_for_npm(
    input_path,
    hdu=1,
    additional_label_names=(
        "source_id", "ra", "dec",
        "phot_rp_mean_mag",
        "phot_g_mean_mag",
        "bp_rp",
        "rv_nb_transits",
        "parallax",
        "parallax_error",
    ))

t = Table(data=data)
t.write(output_path, overwrite=True)
