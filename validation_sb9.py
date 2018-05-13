
"""
Validation of the model predictions compared to other binary catalogues.
"""

import numpy as np
import matplotlib.pyplot as plt
import velociraptor

from astropy.table import Table

sources = Table.read("data/rv-all.fits")

unimodal_catalog = Table.read("data/sb9_matched_by_position.fits")

unimodal_indices = np.nan * np.ones(len(unimodal_catalog))
for i, source_id in enumerate(unimodal_catalog["source_id"]):

    try:
        unimodal_indices[i] = np.where(sources["source_id"] == int(source_id))[0][0]

    except:
        continue

finite = np.isfinite(unimodal_indices)
unimodal_catalog = unimodal_catalog[finite]
unimodal_sources = sources[unimodal_indices[finite].astype(int)]


scalar = 1.0 / (unimodal_catalog["Per"] * (1 - unimodal_catalog["e"]**2)**0.5)
x = unimodal_catalog["K1"] * scalar
y = unimodal_sources["excess_rv_sigma"] * scalar
c = unimodal_sources["p_sbx"]


fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
scat = ax.scatter(x, y, c=c, vmin=0, vmax=1, s=5, cmap="coolwarm_r")
ax.loglog()

ax.set_xlim(10**-4.5, 10**3)
ax.set_ylim(10**-4.5, 10**3)

cbar = plt.colorbar(scat)
cbar.set_label(r"\textrm{binary probability}")
ax.set_xlabel(r"${K}/{P\sqrt{1-e^2}}$")
ax.set_ylabel(r"${\sigma_\textrm{vrad excess}}/{P\sqrt{1-e^2}}$")

fig.tight_layout()
fig.savefig("figures/sb9-comparison.pdf", dpi=150)



x = unimodal_catalog["K1"]
y = unimodal_sources["excess_rv_sigma"]
c = unimodal_sources["p_sbx"]


fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
scat = ax.scatter(x, y, c=c, vmin=0, vmax=1, s=5, cmap="coolwarm_r")

ax.loglog()

ax.set_xlim(10**-0.5, 10**2.5)
ax.set_ylim(10**-0.5, 10**2.5)

cbar = plt.colorbar(scat)
cbar.set_label(r"\textrm{binary probability}")
ax.set_xlabel(r"$K$ $(\textrm{km\,s}^{-1})$")
ax.set_ylabel(r"$\sigma_\textrm{vrad excess}$ $(\textrm{km\,s}^{-1})$")

fig.tight_layout()
fig.savefig("figures/sb9-compare-K-log.pdf", dpi=150)




fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
scat = ax.scatter(x, y, c=c, vmin=0, vmax=1, s=5, cmap="coolwarm_r")

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

cbar = plt.colorbar(scat)
cbar.set_label(r"\textrm{binary probability}")
ax.set_xlabel(r"$K$ $(\textrm{km\,s}^{-1})$")
ax.set_ylabel(r"$\sigma_\textrm{vrad excess}$ $(\textrm{km\,s}^{-1})$")

fig.tight_layout()

fig.savefig("figures/sb9-compare-K.pdf", dpi=150)

is_binary = unimodal_sources["p_sbx"] > 0.5
bins = np.linspace(0, 3000, 50)

fig, ax = plt.subplots()
ax.hist(unimodal_catalog["Per"][is_binary], facecolor="b", bins=bins, alpha=0.5)
ax.hist(unimodal_catalog["Per"][~is_binary], facecolor="r", bins=bins, alpha=0.5)

ax.axvline(682, linestyle=":", c="#666666", lw=1)
ax.axvline(682 * 2, linestyle=":", c="#666666", lw=1)
ax.set_xlabel(r"\textrm{period} $(\textrm{days})$")
ax.set_ylabel(r"\textrm{count}")

fig.tight_layout()

fig.savefig("figures/sb9-period-distribution.pdf", dpi=150)
