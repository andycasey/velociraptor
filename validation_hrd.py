
"""
Validation of the model predictions by making H-R diagram plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import velociraptor
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

from astropy.table import Table

sources = Table.read("data/rv-all.fits")


is_binary = sources["p_sbx"] > 0.5
finite = np.isfinite(sources["bp_rp"] * sources["absolute_g_mag"]) \
       * ((sources["parallax"]/sources["parallax_error"]) > 5) \
       * np.isfinite(sources["radial_velocity"])


fig, axes = plt.subplots(1, 2, figsize=(8, 4))


N_bins = 100
H_all, xedges, yedges = np.histogram2d(
    sources["bp_rp"][finite],
    sources["absolute_g_mag"][finite],
    bins=(N_bins, N_bins))

H_bin, _, __ = np.histogram2d(
    sources["bp_rp"][is_binary * finite],
    sources["absolute_g_mag"][is_binary * finite],
    bins=(xedges, yedges))


H = H_bin/H_all
H[H_all < 3] = np.nan

kwds = dict(
    aspect=np.ptp(xedges)/np.ptp(yedges), 
    extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
)

image = axes[0].imshow(H_all.T, norm=LogNorm(), cmap="Greys", **kwds)

image = axes[1].imshow(H.T, cmap="viridis", **kwds)
cax = fig.add_axes([0.825, 0.85, 0.125, 0.05])

cbar = plt.colorbar(image, cax=cax, orientation="horizontal")
cbar.set_ticks([0, 1])
cbar.set_label(r"\textrm{binary fraction}")

fig.tight_layout()


for ax in axes:
    ax.set_xlabel(r"\textrm{bp-rp}")
    ax.set_ylabel(r"\textrm{absolute G magnitude}")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))


axes[0].text(0.95, 0.95, r"$N = {0}$".format(sum(finite)), 
    transform=axes[0].transAxes,
    horizontalalignment="right", verticalalignment="top")
fig.tight_layout()

fig.savefig("figures/validation-hrd-sbx.pdf", dpi=150)