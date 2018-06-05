
"""
Calculate the completeness of radial velocity measurements as a function of
various properties so that we can gauge when we should use the *lack* of radial
velocity as an indicator of binarity.
"""

import pickle
import numpy as np
from astropy.io import fits
from collections import Counter

N_bins = 50

semilogx = [True, False]
equidensity = [True, False]

completeness = {
    "N_bins": N_bins,
    "semilogx": np.repeat(semilogx, 2),
    "equidensity": np.tile(equidensity, 2),
    "completeness": dict()
}

image = fits.open("data/rv-completeness.fits")
data = image[1].data

has_rv = np.isfinite(data["radial_velocity"])

label_names = (
 'ra',
 'dec',
 'parallax',
 'pmra',
 'pmdec',
 'astrometric_gof_al',
 'astrometric_chi2_al',
 'astrometric_weight_al',
 'astrometric_pseudo_colour',
 'mean_varpi_factor_al',
 'astrometric_matched_observations',
 'visibility_periods_used',
 'astrometric_sigma5d_max',
 'phot_g_mean_flux',
 'phot_g_mean_mag',
 'phot_bp_mean_flux',
 'phot_bp_mean_mag',
 'phot_rp_mean_flux',
 'phot_rp_mean_mag',
 'phot_bp_rp_excess_factor',
 'bp_rp',
 'bp_g',
 'g_rp',
 'radial_velocity',
 'radial_velocity_error',
 'rv_template_teff',
 'rv_template_logg',
 'rv_template_fe_h',
 'l',
 'b',
 'ecl_lon',
 'ecl_lat',
 'teff_val',
 'a_g_val',
 'radius_val',
 'lum_val'
)

for label_name in label_names:

    print(label_name)
    if label_name in completeness["completeness"]: continue

    x = image[1].data[label_name]
    x_min, x_max = (np.nanmin(x), np.nanmax(x))

    plot_bins = np.zeros((4, N_bins), dtype=float)
    plot_x = np.zeros((4, N_bins - 1), dtype=float)
    plot_y = np.zeros_like(plot_x)
    plot_y_err = np.zeros_like(plot_x)

    completeness["completeness"][label_name] = dict()

    for j, _semilogx in enumerate(semilogx):

        for i, _equidensity in enumerate(equidensity):

            if _equidensity:
                p = np.linspace(0, 100, N_bins)
                finite = np.isfinite(x)
                if _semilogx:
                    bins = 10**np.percentile(np.log10(x[finite]), p)

                else:
                    bins = np.percentile(x[finite], p)

            else:
                if _semilogx:
                    space = np.logspace
                    args = (np.log10(x_min), np.log10(x_max), N_bins)

                else:
                    space = np.linspace
                    args = (x_min, x_max, N_bins)

                bins = space(*args)

            centroids = bins[:-1] + 0.5 * np.diff(bins)
            
            indices = np.digitize(x, bins) - 1

            counts = Counter(indices)
            N_per_bin = np.array([counts.get(k, 0) for k in range(N_bins - 1)],
                                 dtype=float)

            counts_finite = Counter(indices[has_rv])
            N_rv_per_bin = np.array([counts_finite.get(k, 0) for k in range(N_bins - 1)],
                                    dtype=float)

            f_rv_per_bin = N_rv_per_bin / N_per_bin
            f_rv_per_bin_err = f_rv_per_bin * np.sqrt(
                (np.sqrt(N_rv_per_bin)/N_rv_per_bin)**2 + \
                (np.sqrt(N_per_bin)/N_per_bin)**2)

            #plot_edges[i] = edges
            index = j * 2 + i
            plot_bins[index, :] = bins
            plot_x[index, :] = centroids
            plot_y[index, :] = f_rv_per_bin
            plot_y_err[index, :] = f_rv_per_bin_err

    completeness["completeness"][label_name]["bins"] = plot_bins
    completeness["completeness"][label_name]["x"] = plot_x
    completeness["completeness"][label_name]["y"] = plot_y
    completeness["completeness"][label_name]["yerr"] = plot_y_err

    with open("rv_completeness.pkl", "wb") as fp:
        pickle.dump(completeness, fp, -1)

