
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
            properties["absolute_{}_mag".format(band)] = \
                  sources["phot_{}_mean_mag".format(band)] \
                + 5 * np.log10(sources["parallax"]/100.0)


        properties["rv_single_epoch_variance"] = sources["radial_velocity_error"]**2 \
                                               * sources["rv_nb_transits"] * np.pi/2.0
        properties["rv_single_epoch_scatter"] = properties["rv_single_epoch_variance"]**0.5

        # Approximate temperature from bp-rp colour
        use_in_fit = np.isfinite(sources["radial_velocity"]) \
                   * (sources["phot_bp_rp_excess_factor"] < 1.5) \
                   * np.isfinite(sources["bp_rp"]) \
                   * np.isfinite(sources["teff_val"]) \
                   * (sources["bp_rp"] < 2.5) \
                   * (sources["bp_rp"] > 0.5)

        x = sources["bp_rp"][use_in_fit]
        y = sources["teff_val"][use_in_fit]

        coeff = np.polyfit(1.0/x, y, 2)

        properties["approx_teff_from_bp_rp"] = np.clip(
            np.polyval(coeff, 1.0/sources["bp_rp"]),
            3500, 8000)

        properties["rv_abs_diff_template_teff"] \
            = np.abs(properties["approx_teff_from_bp_rp"] - sources["rv_template_teff"])

        
        # Astrometric unit weight error
        properties["astrometric_unit_weight_error"] = np.sqrt(
            sources["astrometric_chi2_al"]/(sources["astrometric_n_obs_al"] - 5))
        
        # Photometric variability
        properties["phot_rp_variability"] = np.sqrt(sources["astrometric_n_good_obs_al"]) \
                                          * sources["phot_rp_mean_flux"] \
                                          / sources["phot_rp_mean_flux_error"]

        for label_name in additional_label_names:
            properties[label_name] = sources[label_name]
    
    return properties


data = create_subset_for_npm(
    input_path,
    hdu=1,
    additional_label_names=(
        "source_id", "ra", "dec",
        "phot_rp_mean_mag",
        "bp_rp"
    ))

t = Table(data=data)
t.write(output_path)
