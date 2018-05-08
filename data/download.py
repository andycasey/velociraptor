
"""
Query used to retrieve the radial velocity calibration data from the Gaia
archive.
"""

from astroquery.gaia import Gaia

columns = (
    "source_id",
    "ra", "dec", "l", "b",
    "parallax", "parallax_error",
    "phot_g_mean_mag",
    "phot_g_mean_flux", "phot_g_mean_flux_error",
    "phot_bp_mean_flux", "phot_bp_mean_flux_error", "phot_bp_n_obs",
    "phot_rp_mean_flux", "phot_rp_mean_flux_error", "phot_rp_n_obs",
    "phot_bp_rp_excess_factor",
    "bp_rp", "bp_g", "g_rp",
    "teff_val", "teff_percentile_lower", "teff_percentile_upper",
    "a_g_val", "a_g_percentile_lower", "a_g_percentile_upper",
    "radial_velocity", "radial_velocity_error", "rv_nb_transits",
    "rv_template_teff", "rv_template_logg", "rv_template_fe_h",
    "astrometric_weight_al", "astrometric_gof_al", "astrometric_chi2_al"
)

job = Gaia.launch_job_async("""
    SELECT  {0}
    FROM    gaiadr2.gaia_source
    WHERE   radial_velocity IS NOT NULL
    AND     duplicated_source = 'false'
    AND     rv_nb_transits > 10
    AND     visibility_periods_used > 10
    AND     radial_velocity_error < 20
    AND     MOD(random_index, 10) = 0
    """.format(", ".join(columns)))
subset = job.get_results()
subset.write("rv_floor_cal_subset-result.fits")


job = Gaia.launch_job_async("""
    SELECT  {0}
    FROM    gaiadr2.gaia_source
    WHERE   radial_velocity IS NOT NULL
    AND     duplicated_source = 'false'
    AND     rv_nb_transits > 10
    AND     visibility_periods_used > 10
    AND     radial_velocity_error < 20
    """.format(", ".join(columns)))
sources = job.get_results()
sources.write("rv_floor_cal-result.fits")