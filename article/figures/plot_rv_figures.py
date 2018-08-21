import numpy as np
import os
import pickle
from astropy.io import fits
from astropy.table import Table

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_utils import (mpl_style, plot_binned_statistic)

plt.style.use(mpl_style)


BASE_PATH = "../../"
RELEASE_CANDIDATE_VERSION = 0

MAKE_FIGURES = [
    "rv_hrd_single_star_fraction_hist",
    "rv_hrd_single_star_fraction_scatter",
    "rv_sb9_kalpha_comparison",
    "rv_sb9_k_comparison",
    "rv_sb9_kalpha_comparison_period",
    "rv_sb9_kp_corner",
    "rv_apw_unimodal_k_comparison",
    "rv_apw_unimodal_kalpha_comparison",
    "rv_apw_percentiles_k_comparison",
    "rv_gp_hrd",
    "rv_gp_wrt_params",
    "rv_soubiran_hist",
    "rv_huang_hist"
]

DEFAULT_SEQUENTIAL_CMAP = "viridis"


velociraptor = fits.open(os.path.join(
    BASE_PATH,
    "results", 
    "velociraptor-catalog-rc.{:.0f}.fits".format(RELEASE_CANDIDATE_VERSION)))
velociraptor = velociraptor[1].data

latex_labels = dict(
    bp_rp=r"\textrm{bp - rp}",
    phot_rp_mean_mag=r"\textrm{apparent rp mag}",
    absolute_rp_mag=r"\textrm{absolute rp mag}",
    rv_mu_single=r"$\mu_s$ \textrm{/ km\,s}$^{-1}$",
    rv_sigma_single=r"$\sigma_s$ \textrm{/ km\,s}$^{-1}$"
)


common_kwds = dict([
    ("bp_rp.limits", (-0.25, 6.5)),
    ("phot_rp_mean_mag.limits", (13.0, 2.1)),
    ("absolute_rp_mag.limits", (10.5, -16)),
])

one_to_one_line_kwds = dict(c="#666666", linestyle=":", lw=1, zorder=-1)

def savefig(fig, basename):
    fig.savefig("{}.pdf".format(basename), dpi=300)
    fig.savefig("{}.png".format(basename), dpi=150)
    print("Saved figure {0}.png and {0}.pdf".format(basename))


def cross_match(A_source_ids, B_source_ids):

    A = np.array(A_source_ids, dtype=long)
    B = np.array(B_source_ids, dtype=long)

    ai = np.where(np.in1d(A, B))[0]
    bi = np.where(np.in1d(B, A))[0]
    
    assert len(ai) == len(bi)
    ai = ai[np.argsort(A[ai])]
    bi = bi[np.argsort(B[bi])]

    assert all(A[ai] == B[bi])
    return (ai, bi)



def estimate_K(rv_single_epoch_scatter, rv_mu_single, rv_sigma_single, 
               rv_mu_single_var, rv_sigma_single_var, **kwargs):

    # TODO: Consult with colleagues to see if there is anything more principaled
    #       we could do without requiring much more work.

    K = rv_single_epoch_scatter - rv_mu_single
    K_err = np.sqrt(rv_mu_single_var + rv_sigma_single**2 + rv_sigma_single_var)

    return (K, K_err)


# some metric that is useful for showing "likelihood of binarity" until I decide
# what is best.
log_K_significance = np.log10(
    (velociraptor["rv_single_epoch_scatter"] - velociraptor["rv_mu_single"]) \
  /  velociraptor["rv_sigma_single"])



# Plot the H-R diagram coloured by \tau_{single} as a scatter plot and a 2D hist
figure_name = "rv_hrd_single_star_fraction_hist"
if figure_name in MAKE_FIGURES:

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    xlabel = "bp_rp"
    ylabel = "phot_rp_mean_mag"
    zlabel = "rv_tau_single"

    # Restrict sensibly.
    mask = (velociraptor["absolute_rp_mag"] > -16) \
         * (velociraptor["bp_rp"] < 6.5)

    plot_binned_statistic_kwds = dict(function="mean", vmin=0, vmax=1, bins=100,
                                      xlabel=latex_labels.get(xlabel, xlabel),
                                      cmap=DEFAULT_SEQUENTIAL_CMAP, mask=mask, 
                                      subsample=None, min_entries_per_bin=5)

    for ax, ylabel in zip(axes, ("phot_rp_mean_mag", "absolute_rp_mag")):

        plot_binned_statistic(
            velociraptor[xlabel],
            velociraptor[ylabel],
            velociraptor[zlabel],
            ax=ax, ylabel=latex_labels.get(ylabel, ylabel),
            **plot_binned_statistic_kwds)

        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))

        ax.set_xlim(common_kwds.get("{}.limits".format(xlabel), None))
        ax.set_ylim(common_kwds.get("{}.limits".format(ylabel), None))

    cbar = plt.colorbar(ax.images[0], fraction=0.046, pad=0.04)
    cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")
    fig.tight_layout()

    savefig(fig, figure_name)


figure_name = "rv_hrd_single_star_fraction_scatter"
if figure_name in MAKE_FIGURES:

    subsample = None
    ordered, reverse = True, False

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    xlabel = "bp_rp"
    ylabel = "phot_rp_mean_mag"
    zlabel = "rv_tau_single"

    if subsample is not None:
        idx = np.random.choice(len(velociraptor), subsample, replace=False)
    else:
        idx = np.arange(len(velociraptor))

    if ordered:
        idx = idx[np.argsort(velociraptor[zlabel][idx])]
        if reverse:
            idx = idx[::-1]

    scatter_kwds = dict(vmin=0, vmax=1, cmap=DEFAULT_SEQUENTIAL_CMAP, s=1, 
                        alpha=0.1, c=velociraptor[zlabel][idx])

    for ax, ylabel in zip(axes, ("phot_rp_mean_mag", "absolute_rp_mag")):
        ax.scatter(
            velociraptor[xlabel][idx],
            velociraptor[ylabel][idx],
            **scatter_kwds)

        ax.set_xlabel(latex_labels.get(xlabel, xlabel))
        ax.set_ylabel(latex_labels.get(ylabel, ylabel))

        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
    
        ax.set_xlim(common_kwds.get("{}.limits".format(xlabel), None))
        ax.set_ylim(common_kwds.get("{}.limits".format(ylabel), None))


    if scatter_kwds.get("alpha", 1) == 1:
        collection = ax.collections[0]

    else:
        # Note: if you give alpha in scatter_kwds then you should give
        #       vmin and vmax too otherwise this needs to be updated:
        collection = ax.scatter([np.nanmean(velociraptor[xlabel][idx])],
                                [np.nanmean(velociraptor[ylabel][idx])],
                                c=[np.nanmean(scatter_kwds["c"])],
                                vmin=scatter_kwds["vmin"],
                                vmax=scatter_kwds["vmax"],
                                cmap=scatter_kwds["cmap"],
                                alpha=1.0, s=0)

    cbar = plt.colorbar(collection, fraction=0.046, pad=0.04)
    cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")
    fig.tight_layout()

    savefig(fig, figure_name)




# Cross-match against SB9 catalog only if we have to.
if any(["sb9" in figure_name.lower() for figure_name in MAKE_FIGURES]):

    sb9 = Table.read(os.path.join(BASE_PATH, "data", "sb9_xm_gaia.fits"))

    # remove duplicates.
    sb9 = sb9.group_by("source_id")
    sb9 = sb9[sb9.groups.indices[:-1]]

    assert len(set(sb9["source_id"])) == len(sb9)

    vl_sb9_ids, sb9_ids = cross_match(velociraptor["source_id"], sb9["source_id"])


figure_name = "rv_sb9_kalpha_comparison"
if figure_name in MAKE_FIGURES:

    sort, reverse = (True, False)
    vl_sb9_subset = velociraptor[vl_sb9_ids]

    K_est, K_est_err = estimate_K(vl_sb9_subset["rv_single_epoch_scatter"],
                                  vl_sb9_subset["rv_mu_single"],
                                  vl_sb9_subset["rv_sigma_single"],
                                  vl_sb9_subset["rv_mu_single_var"],
                                  vl_sb9_subset["rv_sigma_single_var"])


    scalar = (1.0 / (sb9["Per"] * (1 - sb9["e"]**2)**0.5))[sb9_ids]

    x = sb9["K1"][sb9_ids] * scalar
    y = K_est * scalar
    yerr = K_est_err * scalar
    c = log_K_significance[vl_sb9_ids]

    if sort:
        idx = np.argsort(c)
        if reverse:
            idx = idx[::-1]
    else:
        idx = np.arange(len(c))

    x, y, yerr, c = (x[idx], y[idx], yerr[idx], c[idx]) 

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    scat = ax.scatter(x, y, c=c, s=10, vmax=1.75)
    ax.errorbar(x, y, yerr=yerr, fmt=None, ecolor="#CCCCCC", zorder=-1, 
                linewidth=1, capsize=0)

    ax.loglog()

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, **one_to_one_line_kwds)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(scat, cax=cax)

    #cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")
    cbar.set_label(r"$\log_{10}\left[\sigma(\textrm{V}_\textrm{R}^{t}) - \mu_s\right] - \log_{10}{\sigma_s}$")
    fig.tight_layout()

    #ax.set_xlabel(r"$K/P\sqrt{1 - e^2}$ \textrm{(SB9)}")
    #ax.set_ylabel(r"$K_\textrm{est} /P\sqrt{1 - e^2}$ \textrm{(this work)}")
    ax.set_xlabel(r"$K_{1} / \alpha$ \textrm{(SB9)}")
    ax.set_ylabel(r"$K_{1,\textrm{est}} / \alpha$ \textrm{(this work)}")

    fig.tight_layout()

    savefig(fig, figure_name)



figure_name = "rv_sb9_k_comparison"
if figure_name in MAKE_FIGURES:

    sort, reverse = (True, False)
    vl_sb9_subset = velociraptor[vl_sb9_ids]

    K_est, K_est_err = estimate_K(vl_sb9_subset["rv_single_epoch_scatter"],
                                  vl_sb9_subset["rv_mu_single"],
                                  vl_sb9_subset["rv_sigma_single"],
                                  vl_sb9_subset["rv_mu_single_var"],
                                  vl_sb9_subset["rv_sigma_single_var"])

    x = sb9["K1"][sb9_ids]
    y = K_est
    yerr = K_est_err
    c = log_K_significance[vl_sb9_ids]

    if sort:
        idx = np.argsort(c)
        if reverse:
            idx = idx[::-1]
    else:
        idx = np.arange(len(c))

    x, y, yerr, c = (x[idx], y[idx], yerr[idx], c[idx]) 

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    scat = ax.scatter(x, y, c=c, s=10, vmax=1.75)
    ax.errorbar(x, y, yerr=yerr, fmt=None, ecolor="#CCCCCC", zorder=-1, 
                linewidth=1, capsize=0)

    ax.loglog()

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, **one_to_one_line_kwds)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(scat, cax=cax)

    #cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")
    cbar.set_label(r"$\log_{10}\left[\sigma(\textrm{V}_\textrm{R}^{t}) - \mu_s\right] - \log_{10}{\sigma_s}$")
    fig.tight_layout()

    #ax.set_xlabel(r"$K/P\sqrt{1 - e^2}$ \textrm{(SB9)}")
    #ax.set_ylabel(r"$K_\textrm{est} /P\sqrt{1 - e^2}$ \textrm{(this work)}")
    ax.set_xlabel(r"$K_{1} \textrm{/ km\,s}^{-1}$ \textrm{(SB9)}")
    ax.set_ylabel(r"$K_{1,\textrm{est}} \textrm{/ km\,s}^{-1}$ \textrm{(this work)}")

    fig.tight_layout()

    savefig(fig, figure_name)




figure_name = "rv_sb9_kalpha_comparison_period"
if figure_name in MAKE_FIGURES:

    sort, reverse = (True, False)
    vl_sb9_subset = velociraptor[vl_sb9_ids]

    K_est, K_est_err = estimate_K(vl_sb9_subset["rv_single_epoch_scatter"],
                                  vl_sb9_subset["rv_mu_single"],
                                  vl_sb9_subset["rv_sigma_single"],
                                  vl_sb9_subset["rv_mu_single_var"],
                                  vl_sb9_subset["rv_sigma_single_var"])


    scalar = (1.0 / (sb9["Per"] * (1 - sb9["e"]**2)**0.5))[sb9_ids]

    x = sb9["K1"][sb9_ids] * scalar
    y = K_est * scalar
    yerr = K_est_err * scalar
    c = np.log10(sb9["Per"][sb9_ids])

    if sort:
        idx = np.argsort(c)
        if reverse:
            idx = idx[::-1]
    else:
        idx = np.arange(len(c))

    x, y, yerr, c = (x[idx], y[idx], yerr[idx], c[idx]) 

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    scat = ax.scatter(x, y, c=c, s=10, vmax=1.75)
    ax.errorbar(x, y, yerr=yerr, fmt=None, ecolor="#CCCCCC", zorder=-1, 
                linewidth=1, capsize=0)

    ax.loglog()

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, **one_to_one_line_kwds)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(scat, cax=cax)

    #cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")
    cbar.set_label(r"$\log_{10}\left(P \textrm{/ days}\right)$")
    fig.tight_layout()

    #ax.set_xlabel(r"$K/P\sqrt{1 - e^2}$ \textrm{(SB9)}")
    #ax.set_ylabel(r"$K_\textrm{est} /P\sqrt{1 - e^2}$ \textrm{(this work)}")
    ax.set_xlabel(r"$K_{1} / \alpha$ \textrm{(SB9)}")
    ax.set_ylabel(r"$K_{1,\textrm{est}} / \alpha$ \textrm{(this work)}")

    fig.tight_layout()

    savefig(fig, figure_name)


figure_name = "rv_sb9_kp_corner"
if figure_name in MAKE_FIGURES:

    sort, reverse = (True, False)

    x = sb9["K1"][sb9_ids]
    y = sb9["Per"][sb9_ids]
    c = velociraptor["rv_tau_single"][vl_sb9_ids]

    if sort:
        idx = np.argsort(c)
        idx = idx[::-1] if reverse else idx

    else:
        idx = np.arange(len(c))


    fig, axes = plt.subplots(2, 2)

    axes[0, 1].set_visible(False)
    axes[1, 0].scatter(x[idx], y[idx], c=c[idx], s=10, vmax=1.75)

    axes[1, 0].loglog()
    axes[1, 0].set_xlabel(r"$P$ \textrm{/ days}")
    axes[1, 0].set_ylabel(r"$K_{1}$ \textrm{/ km\,s}$^{-1}$")

    # TODO: Draw histograms on other axes

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)

    for ax in (axes[0, 0], axes[1, 1]):
        ax.set_xticks([])
        ax.set_yticks([])

    savefig(fig, figure_name)





# Cross-match against APW catalog only if we have to.
if any(["apw_unimodal" in figure_name.lower() for figure_name in MAKE_FIGURES]):

    apw_unimodal = Table.read(os.path.join(
        BASE_PATH, "data", "apw-highK-unimodal-xm-gaia.fits"))

    # remove duplicates.
    apw_unimodal = apw_unimodal.group_by("source_id")
    apw_unimodal = apw_unimodal[apw_unimodal.groups.indices[:-1]]

    assert len(set(apw_unimodal["source_id"])) == len(apw_unimodal)

    vl_apw_um_ids, apw_um_ids = cross_match(velociraptor["source_id"],
                                            apw_unimodal["source_id"])




figure_name = "rv_apw_unimodal_k_comparison"
if figure_name in MAKE_FIGURES:

    sort, reverse = (True, False)

    vl_apwu_subset = velociraptor[vl_apw_um_ids]

    K_est, K_est_err = estimate_K(vl_apwu_subset["rv_single_epoch_scatter"],
                                  vl_apwu_subset["rv_mu_single"],
                                  vl_apwu_subset["rv_sigma_single"],
                                  vl_apwu_subset["rv_mu_single_var"],
                                  vl_apwu_subset["rv_sigma_single_var"])

    x = apw_unimodal["K"][apw_um_ids]
    xerr = apw_unimodal["K_err"][apw_um_ids]
    y = K_est
    yerr = K_est_err
    c = vl_apwu_subset["rv_tau_single"]

    if sort:
        idx = np.argsort(c)
        idx = idx[::-1] if reverse else idx

    else:
        idx = np.arange(len(c))

    x, y, xerr, yerr, c = (x[idx], y[idx], xerr[idx], yerr[idx], c[idx])


    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    scat = ax.scatter(x, y, c=c, s=10, vmin=0, vmax=1)
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=None, ecolor="#cccccc", lw=1,
                zorder=-1)
    ax.loglog()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(scat, cax=cax)

    cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")
    #cbar.set_label(r"$\log_{10}\left(P \textrm{/ days}\right)$")

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, **one_to_one_line_kwds)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_xlabel(r"$K$ \textrm{/ km\,s}$^{-1}$ \textrm{(Price-Whelan et al. 2017)}")
    ax.set_ylabel(r"$K$ \textrm{/ km\,s}$^{-1}$ \textrm{(this work)}")

    fig.tight_layout()

    savefig(fig, figure_name)


figure_name = "rv_apw_unimodal_kalpha_comparison"
if figure_name in MAKE_FIGURES:

    sort, reverse = (True, False)

    vl_apwu_subset = velociraptor[vl_apw_um_ids]

    K_est, K_est_err = estimate_K(vl_apwu_subset["rv_single_epoch_scatter"],
                                  vl_apwu_subset["rv_mu_single"],
                                  vl_apwu_subset["rv_sigma_single"],
                                  vl_apwu_subset["rv_mu_single_var"],
                                  vl_apwu_subset["rv_sigma_single_var"])

    scalar = 1.0/(apw_unimodal["P"] * np.sqrt(1 - apw_unimodal["e"]**2))[apw_um_ids]

    x = apw_unimodal["K"][apw_um_ids] * scalar
    xerr = apw_unimodal["K_err"][apw_um_ids] * scalar
    y = K_est * scalar
    yerr = K_est_err * scalar
    c = vl_apwu_subset["rv_tau_single"]

    if sort:
        idx = np.argsort(c)
        idx = idx[::-1] if reverse else idx

    else:
        idx = np.arange(len(c))

    x, y, xerr, yerr, c = (x[idx], y[idx], xerr[idx], yerr[idx], c[idx])

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    scat = ax.scatter(x, y, c=c, s=10, vmin=0, vmax=1)
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=None, ecolor="#cccccc", lw=1,
                zorder=-1)
    ax.loglog()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(scat, cax=cax)

    cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")
    #cbar.set_label(r"$\log_{10}\left(P \textrm{/ days}\right)$")

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, **one_to_one_line_kwds)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_xlabel(r"$K / \alpha$ \textrm{(Price-Whelan et al. 2017)}")
    ax.set_ylabel(r"$K / \alpha$ \textrm{(this work)}")

    fig.tight_layout()

    savefig(fig, figure_name)




# Cross-match against APW catalog only if we have to.
if any(["apw_percentiles" in figure_name.lower() for figure_name in MAKE_FIGURES]):

    apw_percentiles = Table.read(os.path.join(
        BASE_PATH, "data", "apw-lnK-percentiles-xm-gaia.fits"))

    # remove duplicates.
    apw_percentiles = apw_percentiles.group_by("source_id")
    apw_percentiles = apw_percentiles[apw_percentiles.groups.indices[:-1]]

    assert len(set(apw_percentiles["source_id"])) == len(apw_percentiles)

    vl_apw_lnk_ids, apw_lnk_ids = cross_match(velociraptor["source_id"],
                                              apw_percentiles["source_id"])


figure_name = "rv_apw_percentiles_k_comparison"
if figure_name in MAKE_FIGURES:

    sort, reverse = (True, False)

    vl_apwp_subset = velociraptor[vl_apw_lnk_ids]

    K_est, K_est_err = estimate_K(vl_apwp_subset["rv_single_epoch_scatter"],
                                  vl_apwp_subset["rv_mu_single"],
                                  vl_apwp_subset["rv_sigma_single"],
                                  vl_apwp_subset["rv_mu_single_var"],
                                  vl_apwp_subset["rv_sigma_single_var"])

    x = np.exp(apw_percentiles["lnK_per_1"][apw_lnk_ids])
    y = K_est
    yerr = K_est_err
    c = vl_apwp_subset["rv_tau_single"]

    if sort:
        idx = np.argsort(c)
        idx = idx[::-1] if reverse else idx
    else:
        idx = np.arange(len(c))

    x, y, yerr, c = (x[idx], y[idx], yerr[idx], c[idx])

    fig, ax = plt.subplots(figsize=(8, 7))
    scat = ax.scatter(x, y, c=c, s=1)
    ax.errorbar(x, y, yerr=yerr, fmt=None, ecolor="#cccccc", lw=1, zorder=-1)

    ax.loglog()

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, **one_to_one_line_kwds)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_xlabel(r"$K_\textrm{1\%}$ \textrm{/ km\,s}$^{-1}$ \textrm{(Price-Whelan et al. 2017)}")
    ax.set_ylabel(r"$K$ \textrm{/ km\,s}$^{-1}$ \textrm{(this work)}")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(scat, cax=cax)

    cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")


    fig.tight_layout()

    savefig(fig, figure_name)


figure_name = "rv_gp_hrd"
if figure_name in MAKE_FIGURES:
    
    xlabel = "bp_rp"
    ylabel = "phot_rp_mean_mag"
    zlabels = ("rv_mu_single", "rv_sigma_single",)
    #           "rv_mu_multiple", "rv_sigma_multiple")

    K, M = (2, len(zlabels))
    fig, axes = plt.subplots(M, K, figsize=(4 * K + 1, 4 * M))

    mask = (velociraptor["absolute_rp_mag"] > -5) \
         * (velociraptor["bp_rp"] < 4)

    limits = dict(bp_rp=(-0.25, 4),
                  phot_rp_mean_mag=(13, 6),
                  absolute_rp_mag=(10, -5))

    plot_binned_statistic_kwds = dict(function="median", bins=100,
                                      xlabel=latex_labels.get(xlabel, xlabel),
                                      cmap=DEFAULT_SEQUENTIAL_CMAP, mask=mask, 
                                      subsample=100000, min_entries_per_bin=5)

    for ax_row, zlabel in zip(axes, zlabels):

        for ax, ylabel in zip(ax_row, ("phot_rp_mean_mag", "absolute_rp_mag")):

            plot_binned_statistic(
                velociraptor[xlabel],
                velociraptor[ylabel],
                velociraptor[zlabel],
                ax=ax, ylabel=latex_labels.get(ylabel, ylabel),
                **plot_binned_statistic_kwds)

            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_locator(MaxNLocator(6))

            ax.set_xlim(limits.get(xlabel, None))
            ax.set_ylim(limits.get(ylabel, None))

            ax.set_title(latex_labels.get(zlabel, zlabel))

        #cbar = plt.colorbar(ax.images[0], fraction=0.046, pad=0.04)
        #cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")

    for ax in np.array(axes).flatten():
        ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

    fig.tight_layout()
    savefig(fig, figure_name)


figure_name = "rv_gp_wrt_params"
if figure_name in MAKE_FIGURES:
    
    from matplotlib.colors import LogNorm

    xlabels = ("bp_rp",
               "phot_rp_mean_mag",
               "absolute_rp_mag")

    ylabels = ("rv_mu_single", "rv_sigma_single")
    
    K, M = (len(xlabels), len(ylabels))
    fig, axes = plt.subplots(M, K, figsize=(4 * K, 4 * M))

    mask = np.isfinite(velociraptor["rv_single_epoch_scatter"])

    plot_binned_statistic_kwds = dict(function="count", bins=250,
                                      cmap="Blues", mask=mask, 
                                      subsample=None, min_entries_per_bin=1,
                                      norm=LogNorm())

    for ax_row, xlabel in zip(axes.T, xlabels):

        for ax, ylabel in zip(ax_row, ylabels):

            plot_binned_statistic(
                velociraptor[xlabel],
                velociraptor[ylabel],
                velociraptor[xlabel],
                ax=ax, 
                xlabel=latex_labels.get(xlabel, xlabel),
                ylabel=latex_labels.get(ylabel, ylabel),
                **plot_binned_statistic_kwds)

            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_locator(MaxNLocator(6))

            ax.set_xlim(np.sort(common_kwds.get("{}.limits".format(xlabel), None)))
            ax.set_ylim(0, max(ax.get_ylim()))
        
        #cbar = plt.colorbar(ax.images[0], fraction=0.046, pad=0.04)
        #cbar.set_label(r"\textrm{single star fraction} $\tau_\textrm{single}$")

    for ax in np.array(axes).flatten():
        ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

    fig.tight_layout()
    savefig(fig, figure_name)




# Cross-match against Soubiran catalog only if we have to.
if any(["soubiran" in figure_name.lower() for figure_name in MAKE_FIGURES]):

    soubiran = Table.read(os.path.join(
        BASE_PATH, "data", "soubiran-2013-xm-gaia.fits"))

    # remove duplicates.
    soubiran = soubiran.group_by("source_id")
    soubiran = soubiran[soubiran.groups.indices[:-1]]

    assert len(set(soubiran["source_id"])) == len(soubiran)

    vl_soubiran_ids, soubiran_ids = cross_match(velociraptor["source_id"],
                                                soubiran["source_id"])


figure_name = "rv_soubiran_hist"
if figure_name in MAKE_FIGURES:

    vl_soubiran_subset = velociraptor[vl_soubiran_ids]
    K_est, K_est_err = estimate_K(vl_soubiran_subset["rv_single_epoch_scatter"],
                                  vl_soubiran_subset["rv_mu_single"],
                                  vl_soubiran_subset["rv_sigma_single"],
                                  vl_soubiran_subset["rv_mu_single_var"],
                                  vl_soubiran_subset["rv_sigma_single_var"])


    fig, axes = plt.subplots(2, 1, figsize=(4, 8))

    axes[0].hist(K_est[np.isfinite(K_est)], bins=100)

    axes[0].set_xlabel(r"$K_{est}$ \textrm{/ km\,s}$^{-1}$")
    axes[0].set_ylabel(r"\textrm{count}")
    
    axes[1].semilogx()
    positive = K_est > 0
    
    axes[1].scatter(K_est[positive],
                    vl_soubiran_subset["rv_tau_single"][positive], s=10)
    
    xlim = axes[1].get_xlim()
    axes[1].errorbar(K_est[positive], 
                     vl_soubiran_subset["rv_tau_single"][positive],
                     xerr=np.min([K_est_err[positive], K_est[positive] - 1e-4], axis=0),
                     fmt=None, ecolor="#CCCCCC", zorder=-1, 
                     linewidth=1, capsize=0)
    axes[1].set_xlim(xlim)

    axes[1].set_xlabel(r"$K_{est} \textrm{ / km\,s}^{-1}$")
    axes[1].set_ylabel(r"\textrm{single star fraction} $\tau_\textrm{single}$")

    for ax in np.array(axes).flatten():
        ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

    fig.tight_layout()

    notable = (vl_soubiran_subset["rv_tau_single"] < 0.6) \
            * (K_est > 0)

    savefig(fig, figure_name)




# Cross-match against Soubiran catalog only if we have to.
if any(["huang" in figure_name.lower() for figure_name in MAKE_FIGURES]):

    huang = Table.read(os.path.join(
        BASE_PATH, "data", "huang-apogee-rv-standards-xm-gaia.fits"))

    # remove duplicates.
    huang = huang.group_by("source_id")
    huang = huang[huang.groups.indices[:-1]]

    assert len(set(huang["source_id"])) == len(huang)

    vl_huang_ids, huang_ids = cross_match(velociraptor["source_id"],
                                          huang["source_id"])


figure_name = "rv_huang_hist"
if figure_name in MAKE_FIGURES:

    vl_huang_subset = velociraptor[vl_huang_ids]
    K_est, K_est_err = estimate_K(vl_huang_subset["rv_single_epoch_scatter"],
                                  vl_huang_subset["rv_mu_single"],
                                  vl_huang_subset["rv_sigma_single"],
                                  vl_huang_subset["rv_mu_single_var"],
                                  vl_huang_subset["rv_sigma_single_var"])


    fig, axes = plt.subplots(2, 1, figsize=(4, 8))

    v = log_K_significance[vl_huang_ids]
    sb9_v = log_K_significance[vl_sb9_ids]
    axes[0].hist(v[np.isfinite(v)], bins=100, normed=True, alpha=0.5)
    axes[0].hist(sb9_v[np.isfinite(sb9_v)], bins=100, facecolor="tab:green", alpha=0.5, normed=True)
    axes[0].set_xlabel(r"$\log_{10}\left[\sigma(\textrm{V}_\textrm{R}^{t}) - \mu_s\right] - \log_{10}{\sigma_s}$")

    axes[0].set_yticks([])

    axes[1].scatter(v, K_est, s=10)
    
    axes[1].errorbar(v, K_est,
                     yerr=np.min([K_est_err, K_est - 1e-4], axis=0),
                     fmt=None, ecolor="#CCCCCC", zorder=-1, 
                     linewidth=1, capsize=0)

    axes[1].set_ylabel(r"$K_{est} \textrm{ / km\,s}^{-1}$")
    axes[1].set_xlabel(r"$\log_{10}\left[\sigma(\textrm{V}_\textrm{R}^{t}) - \mu_s\right] - \log_{10}{\sigma_s}$")
    
    for ax in np.array(axes).flatten():
        ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

    fig.tight_layout()

    notable = (vl_huang_subset["rv_tau_single"] < 0.6) \
            * (K_est > 0)

    savefig(fig, figure_name)