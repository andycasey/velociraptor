
"""
Plot the completeness of radial velocity determinations of various properties
so that we can gauge when we should use the lack of radial velocity as an
indicator of binarity.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from collections import Counter
from matplotlib.ticker import MaxNLocator

from mpl_utils import mpl_style

plt.style.use(mpl_style)

image = fits.open("data/rv-completeness.fits")
data = image[1].data

has_rv = np.isfinite(data["radial_velocity"])

N_bins = 50
semilogy = True

label_names = [
    ("phot_rp_mean_mag", r"\textrm{apparent rp magnitude}", False),
    ("phot_rp_mean_flux", r"\textrm{apparent rp flux}", True),
    ("phot_bp_mean_flux", r"\textrm{apparent bp flux}", True),
    ("phot_bp_mean_mag", r"\textrm{apparent bp magnitude}", False),
    ("bp_rp", r"\textrm{bp - rp}", False),
    ("phot_bp_rp_excess_factor", r"\textrm{bp-rp excess factor}", False),
    ("teff_val", r"$T_{\rm eff}$", False),

]



for label_name, xlabel, semilogx in label_names:

    x = image[1].data[label_name]
    x_min, x_max = (np.nanmin(x), np.nanmax(x))

    plot_bins = np.zeros((2, N_bins), dtype=float)
    plot_x = np.zeros((2, N_bins - 1), dtype=float)
    plot_y = np.zeros_like(plot_x)
    plot_y_err = np.zeros_like(plot_x)

    plot_edges = np.zeros((2, 2 * N_bins), dtype=float)

    equidensity_bins = (True, False)

    for i, equidensity in enumerate(equidensity_bins):

        if equidensity:
            p = np.linspace(0, 100, N_bins)
            finite = np.isfinite(x)
            if semilogx:
                bins = 10**np.percentile(np.log10(x[finite]), p)

            else:
                bins = np.percentile(x[finite], p)

        else:
            if semilogx:
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
        plot_bins[i] = bins
        plot_x[i] = centroids
        plot_y[i] = f_rv_per_bin
        plot_y_err[i] = f_rv_per_bin_err


    K = len(equidensity_bins)
    fig, axes = plt.subplots(1, K, figsize=(6 * K, 6))

    for i, (ax, equidensity) in enumerate(zip(axes, equidensity_bins)):

        bins = plot_x[i]
        edges = np.array(bins).repeat(2)[1:]

        xstep = np.repeat((bins[1:] - bins[:-1]), 2)
        xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
        xstep = np.append(edges, edges.max() + xstep[-1]) - xstep/2.0

        xx = xstep
        yy = np.array(plot_y[i]).repeat(2)

        if semilogx:
            xc = 10**(np.log10(xx)[::2] + 0.5 * np.diff(np.log10(xx))[::2])
        else:
            xc = plot_x[i]

        _ = ax.plot(xx, yy, "-")
        ax.errorbar(xc, plot_y[i], plot_y_err[i], fmt=None, capsize=0,
            ecolor=_[0].get_color())

        title = r"\textrm{equi-density bins}" if equidensity \
                                              else r"\textrm{equi-spaced bins}"
        ax.set_title(title)

        if semilogx:
            ax.semilogx()
        else:
            ax.xaxis.set_major_locator(MaxNLocator(6))

        if semilogy:
            ax.semilogy()
            ax.set_ylim(ax.get_ylim()[0], 1)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(6))

        ax.set_xlim(xx.min() )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"\textrm{fraction of stars with radial velocities}")


    fig.tight_layout()
    fig.savefig("figures/rv_completeness_{}.pdf".format(label_name), dpi=300)

    print("Finished {}".format(label_name))
