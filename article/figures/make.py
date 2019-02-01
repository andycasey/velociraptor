
"""
Produce figures for the Velociraptor project.


Science verification
--------------------

[X] 1. PDF of known single stars (Soubiran) and binary stars (SB9).

[X] 2. SB9 mass versus period, coloured by tau_single_rv.

[X] 3. SB9 mass versus period, coloured by tau_single_ast.

[X] 4. SB9 RV semi-amplitude vs our estimated RV semi-amplitude, w.r.t alpha, coloured by period.

[X] 5. Like #4, but coloured by other things: eccentricity, etc.


Astrophysics
------------

1.  Main-sequence (bp rp vs absolute g-mag), coloured by median single star 
    probability for each panel: RV, astrometry, joint.

2. 



"""

import os
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from mpl_utils import mpl_style

plt.style.use(mpl_style)


BASE_PATH = "../../"
try:
    velociraptor

except NameError:
    velociraptor = fits.open("../../results/die-hard-subset.fits")[1].data

else:
    print("Warning: using pre-loaded velociraptor catalog")



cmap_binary = matplotlib.cm.coolwarm


def cross_match(A_source_ids, B_source_ids):

    A = np.array(A_source_ids, dtype=np.long)
    B = np.array(B_source_ids, dtype=np.long)

    ai = np.where(np.in1d(A, B))[0]
    bi = np.where(np.in1d(B, A))[0]
    
    return (ai[np.argsort(A[ai])], bi[np.argsort(B[bi])])




def _literature_multiplicity_pdfs(source_ids, pdfs,
                                  soubiran_catalog, sb9_catalog, pdf_idx):
    
    # Cross-match our source ids with literature.
    vl_sb9_ids, sb9_ids = cross_match(source_ids, sb9_catalog["source_id"])
    vl_soubiran_ids, soubiran_ids = cross_match(source_ids, soubiran_catalog["source_id"])

    # Build up PDF density.
    pdf_soubiran = pdfs[pdf_idx, vl_soubiran_ids]
    pdf_sb9 = pdfs[pdf_idx, vl_sb9_ids]

    return (pdf_soubiran, pdf_sb9)


def plot_literature_multiplicity_pdf(source_ids, pdfs, 
                                     soubiran_catalog, sb9_catalog, 
                                     pdf_idx=0, colors=None,
                                     **kwargs):

    pdf_soubiran, pdf_sb9 = _literature_multiplicity_pdfs(source_ids, pdfs,
                                                          soubiran_catalog,
                                                          sb9_catalog,
                                                          pdf_idx=pdf_idx)

    bins = np.linspace(0, 1, 21)

    if colors is None:
        colors = [cmap_binary(255), cmap_binary(0)]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.hist([pdf_soubiran, pdf_sb9][::-1], bins=bins, 
        label=[
            r"$\textrm{Single stars (Soubiran et al. 2009)}$",
            r"$\textrm{Binary stars (Pourbaix et al. 2004)}$"
        ][::-1],
        histtype="barstacked", color=colors[::-1])

    plt.legend(frameon=False)
    ax.set_xlabel(r"$p(\textrm{single}|\sigma_{vr})$")

    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(6))

    fig.tight_layout()

    return fig



def plot_literature_multiplicity_classifications(source_ids, pdfs,
                                                 soubiran_catalog, sb9_catalog,
                                                 colors=None,
                                                 pdf_idx=0, **kwargs):

    pdf_soubiran, pdf_sb9 = _literature_multiplicity_pdfs(source_ids, pdfs,
                                                          soubiran_catalog,
                                                          sb9_catalog,
                                                          pdf_idx=pdf_idx)

    soubiran_class = np.round(np.mean(pdf_soubiran, axis=1)).astype(int)
    sb9_class = np.round(np.mean(pdf_sb9, axis=1)).astype(int)

    fig, ax = plt.subplots(figsize=(5, 2.5))

    S_single = np.sum(soubiran_class == 1)
    S_binary = np.sum(soubiran_class == 0)

    SB9_single = np.sum(sb9_class == 1)
    SB9_binary = np.sum(sb9_class == 0)

    if colors is None:
        colors = [cmap_binary(255), cmap_binary(0)]

    ax.barh(0, S_single, facecolor=colors[0], label=r"$\textrm{Classified as single}$")
    ax.barh(0, S_binary, facecolor=colors[1], left=S_single)

    ax.barh(1, SB9_binary, facecolor=colors[1], label=r"$\textrm{Classified as binary}$")
    ax.barh(1, SB9_single, facecolor=colors[0], left=SB9_binary)

    plt.legend(frameon=False)


    ax.set_yticks([0, 1])
    ax.tick_params(axis="y", pad=60)
    ax.set_yticklabels([
        r"$\textrm{Known single stars}$" + "\n" + \
        r"$\textrm{(Soubiran et al. 2009)}$",
        r"$\textrm{Known binary stars}$" + "\n" + \
        r"$\textrm{(Pourbaix et al. 2004)}$"
    ], horizontalalignment="center")


    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlabel(r"$\textrm{Classifications}$")
    fig.tight_layout()

    return fig


def literature_binary_properties(velociraptor, sb9_catalog, x_label="Per",
                                 y_label="K1", z_label="joint_confidence",
                                 errors=True, log_axes=True, latex_labels=None):

    vl_sb9_ids, sb9_ids = cross_match(velociraptor["source_id"], sb9_catalog["source_id"])

    latex_labels_ = dict(Per=r"$\textrm{period}$ $\textrm{/\,days}$",
                         K1=r"$\textrm{radial velocity semi-amplitude}$ $\textrm{/\,km\,s}^{-1}$",
                         e=r"$\textrm{eccentricity}$",
                         rv_confidence=r"$\textrm{Single star confidence (rv only)}$",
                         ast_confidence=r"$\textrm{Single star confidence (ast only)}$",
                         joint_confidence=r"$\textrm{Single star confidence (joint)}$")
    latex_labels_.update(latex_labels or dict())

    x = sb9_catalog[x_label][sb9_ids]
    y = sb9_catalog[y_label][sb9_ids]
    if z_label in velociraptor.dtype.names:
        z = velociraptor[z_label][vl_sb9_ids]
    else:
        z = sb9_catalog[z_label][sb_9_ids]


    fig, ax = plt.subplots(figsize=(6.18, 5.0))

    kwds = dict(s=15, cmap=cmap_binary)
    scat = ax.scatter(x, y, c=z, **kwds)
    if errors:
        ax.errorbar(x, y,
                    xerr=sb9_catalog[f"e_{x_label}"][sb9_ids],
                    yerr=sb9_catalog[f"e_{y_label}"][sb9_ids],
                    fmt=None, zorder=-1, c="#666666", linewidth=0.5)

    if log_axes is True:
        ax.loglog()
    elif isinstance(log_axes, (tuple, list)):
        if log_axes[0]:
            ax.semilogx()
        if log_axes[1]:
            ax.semilogy()

    ax.set_xlabel(latex_labels_.get(x_label, x_label))
    ax.set_ylabel(latex_labels_.get(y_label, y_label))

    cbar = plt.colorbar(scat)
    cbar.set_label(latex_labels_.get(z_label, z_label))

    fig.tight_layout()

    return fig


def plot_semi_amplitude_wrt_literature(velociraptor, sb9_catalog, 
                                       scale=True, loglog=True, z_log=False,
                                       z_label="joint_confidence"):

    vl_sb9_ids, sb9_ids = cross_match(velociraptor["source_id"], sb9_catalog["source_id"])

    K_est = velociraptor["rv_excess"][vl_sb9_ids]
    K_err = velociraptor["rv_excess_var"][vl_sb9_ids]**0.5

    if scale:
        scalar = (1.0 / (sb9_catalog["Per"] * (1 - sb9_catalog["e"]**2)**0.5))[sb9_ids]
    else:
        scalar = 1.0

    y = K_est * scalar
    yerr = K_err * scalar

    xp = np.array([
        sb9_catalog["K1"][sb9_ids] * scalar,
        sb9_catalog["K2"][sb9_ids] * scalar
    ])

    xperr = np.array([
        sb9_catalog["e_K1"][sb9_ids] * scalar,
        sb9_catalog["e_K2"][sb9_ids] * scalar
    ])

    """
    diff = np.abs(xp - y)
    diff[~np.isfinite(diff)] = np.inf
    idx = np.nanargmin(diff, axis=0)
    x = xp[idx, np.arange(y.size)]
    xerr = xperr[idx, np.arange(y.size)]
    """
    x, xerr = xp[0], xperr[0]

    if z_label in velociraptor.dtype.names:
        c = velociraptor[z_label][vl_sb9_ids]

    else:
        c = sb9_catalog[z_label][sb9_ids]

    if z_log:
        c = np.log10(c)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    scat = ax.scatter(x, y, c=c, cmap=cmap_binary, s=15, rasterized=True)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="#CCCCCC", 
                zorder=-1, linewidth=1, capsize=0)

    if loglog:
        ax.loglog()

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))

    kwds = dict(c="#666666", linestyle=":", zorder=-1, linewidth=0.5)
    ax.plot(limits, limits, **kwds)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    latex_labels_ = dict(Per=r"$\textrm{period}$ $\textrm{/\,days}$",
                         K1=r"$\textrm{radial velocity semi-amplitude}$ $\textrm{/\,km\,s}^{-1}$",
                         e=r"$\textrm{eccentricity}$",
                         rv_confidence=r"$\textrm{Single star confidence (rv only)}$",
                         ast_confidence=r"$\textrm{Single star confidence (ast only)}$",
                         joint_confidence=r"$\textrm{Single star confidence (joint)}$",
                         rv_nb_transits=r"$\textrm{number of radial velocity transits}$")

    cbar = plt.colorbar(scat)

    cbar.set_label(latex_labels_.get(z_label, z_label))
    fig.tight_layout()

    if scale:
        ax.set_xlabel(r"$K_{1} / \sqrt{P(1-e^2)}$ \textrm{(Pourbaix et al. 2004)}")
        ax.set_ylabel(r"$K_{1} / \sqrt{P(1-e^2)}$ \textrm{(this work)}")

    else:        
        ax.set_xlabel(r"$K_{1}\,/\,\textrm{km\,s}^{-1}$ \textrm{(Pourbaix et al. 2004)}")
        ax.set_ylabel(r"$K_{1}\,/\,\textrm{km\,s}^{-1}$ \textrm{(this work)}")

    fig.tight_layout()

    return fig


sb9 = Table.read(os.path.join(BASE_PATH, "data", "sb9_xm_gaia.fits"))
sb9 = sb9.group_by("source_id")
sb9 = sb9[sb9.groups.indices[:-1]]
sb9_mask = (sb9["f_K1"] != ">") \
         * (sb9["f_T0"] == 0) \
         * (sb9["Grade"] > 0) \
         * (sb9["f_omega"] != "a") \
         * (sb9["o_K1"] > 0)
sb9 = sb9[sb9_mask]

soubiran = Table.read(os.path.join(
    BASE_PATH, "data", "soubiran-2013-xm-gaia.fits"))

soubiran = soubiran.group_by("source_id")
soubiran = soubiran[soubiran.groups.indices[:-1]]

velociraptor_source_ids = np.memmap("../../results/die-hard-subset.sources.memmap",
                                    mode="r", dtype=">i8")

velociraptor_pdf = np.memmap("../../results/die-hard-subset.pdf.memmap",
                             mode="r", dtype=np.float32,
                             shape=(3, len(velociraptor_source_ids), 100))


model_names = ["ast", "rv", "joint"]


fig = plot_semi_amplitude_wrt_literature(velociraptor, sb9, z_label="Per", 
                                         scale=False, loglog=False, z_log=True)


fig = plot_semi_amplitude_wrt_literature(velociraptor, sb9, z_label="e", 
                                         scale=False, loglog=False)

fig = plot_semi_amplitude_wrt_literature(velociraptor, sb9, scale=False, loglog=False)


fig = plot_semi_amplitude_wrt_literature(velociraptor, sb9, 
                                         z_label="rv_nb_transits",
                                         scale=False, loglog=False)

fig = literature_binary_properties(velociraptor, sb9)


for i, model_name in enumerate(model_names):

    fig = plot_literature_multiplicity_pdf(
            velociraptor_source_ids, velociraptor_pdf,
            soubiran, sb9, pdf_idx=i)

for i, model_name in enumerate(model_names):
    fig_classes = plot_literature_multiplicity_classifications(
                    velociraptor_source_ids, velociraptor_pdf,
                    soubiran, sb9,pdf_idx=i)
