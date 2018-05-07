
"""
Query used to retrieve the radial velocity calibration data from the Gaia
archive.
"""

from astroquery.gaia import Gaia

job = Gaia.launch_job_async("""
    SELECT  source_id,
            parallax, parallax_error,
            phot_g_mean_mag,
            phot_g_mean_flux, phot_g_mean_flux_error,
            phot_bp_mean_flux, phot_bp_mean_flux_error, phot_bp_n_obs,
            phot_rp_mean_flux, phot_rp_mean_flux_error, phot_rp_n_obs,
            phot_bp_rp_excess_factor,
            bp_rp, bp_g, g_rp,
            radial_velocity, radial_velocity_error, rv_nb_transits,
            astrometric_weight_al, astrometric_gof_al, astrometric_chi2_al
    FROM    gaiadr2.gaia_source
    WHERE   radial_velocity IS NOT NULL
    AND     duplicated_source = 'false'
    AND     rv_nb_transits > 10
    AND     visibility_periods_used > 10
    AND     radial_velocity_error < 20
    """)
results = job.get_results()
results.write("rv_floor_cal-result.fits")