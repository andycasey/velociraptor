
"""
Selection and analysis of SB2 type binary stars.

See notebooks/sb2.ipynb for inspiration.

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.io import fits

import velociraptor
from mpl_utils import mpl_style
plt.style.use(mpl_style)


data = fits.open("data/gaia-sources-for-npm.fits")[1].data


# Apply ranges
valid_ranges = dict(
    bp_rp=(1.2, 2.3),
    phot_rp_mean_mag=(10, 12.2))


def get_rv_completeness(x, semilogx, equidensity, N_bins,
                        x_min=None, x_max=None):

    y = data["radial_velocity"]
    
    x_finite = np.isfinite(x)
    y_finite = np.isfinite(y)
    if x_min is None:
        x_min = np.min(x[x_finite])
    if x_max is None:
        x_max = np.max(x[x_finite])
    
    mask = (x_max >= x) * (x >= x_min)
    x_finite *= mask
    y_finite *= mask
    
    p = np.linspace(0, 100, N_bins)
    if equidensity and semilogx:
        bins = 10**np.percentile(np.log10(x[x_finite]), p)
        
    elif equidensity and not semilogx:
        bins = np.percentile(x[x_finite], p)
    
    elif not equidensity and semilogx:
        bins = np.logspace(np.log10(x_min), np.log10(x_max), N_bins)
    
    elif not equidensity and not semilogx:
        bins = np.linspace(x_min, x_max, N_bins)
    
    numerator, _ = np.histogram(x[x_finite * y_finite], bins=bins)
    denominator, _ = np.histogram(x[x_finite], bins=bins)
    f = numerator/denominator.astype(float)
    
    # Pretty sure this ~has~ to be wrong.
    f_err = f / np.diff(bins) * np.sqrt(
            (np.sqrt(numerator)/numerator)**2 + \
            (np.sqrt(denominator)/denominator)**2)

    return (bins, f, f_err, numerator, denominator)


def plot_rv_completeness(x, latex_label_name, semilogx, equidensity, N_bins, 
                         ax=None, title=None, x_min=None, x_max=None, 
                         valid_xrange=None, snap_xrange_to_nearest_bin_edge=False,
                         **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = ax.figure
        
    bins, f, f_err, num, den = get_rv_completeness(
        x, semilogx=semilogx, equidensity=equidensity,
        N_bins=N_bins, x_min=x_min, x_max=x_max)
    
    x = np.hstack([bins[0], np.repeat(bins[1:-1], 2), bins[-1]])
    x_mid = bins[:-1] + 0.5 * np.diff(bins)
    
    y = np.array(f).repeat(2)
    
    kwds = dict(linestyle="-", marker="None")
    kwds.update(kwargs)
    
    line = ax.plot(x, y, **kwds)
    ax.errorbar(x_mid, f, yerr=f_err, fmt="none", c=line[0].get_color())
    
    ax.set_ylabel(r"\textrm{fraction of sources with radial velocity}")
    ax.set_xlabel(latex_label_name)
    
    if semilogx:
        ax.semilogx()
        
    if valid_xrange is not None:
        lower, upper = valid_xrange
        if snap_xrange_to_nearest_bin_edge:
            mask = (bins >= lower) * (bins <= upper)
            lower = bins[mask][0]
            upper = bins[mask][-1]

            print("Updated valid range for {} is ({:.2f}, {:.2f})".format(
                latex_label_name, lower, upper))

        ax.axvspan(lower, upper, facecolor="#dddddd", edgecolor="None", zorder=-1)
    
    if x_min is not None:
        ax.set_xlim(x_min, ax.get_xlim()[1])
    if x_max is not None:
        ax.set_xlim(ax.get_xlim()[0], x_max)


    return fig


label_names = ("bp_rp", "phot_rp_mean_mag")
latex_label_names = dict(bp_rp=r"\textrm{bp - rp}",
                         phot_rp_mean_mag=r"\textrm{apparent rp magnitude}")
K = len(label_names)

fig, axes = plt.subplots(1, K, figsize=(10, 5))

common_kwds = dict(N_bins=30, semilogx=False, equidensity=False,
                   snap_xrange_to_nearest_bin_edge=True)

plot_rv_completeness(
    data["bp_rp"], 
    latex_label_name=r"\textrm{bp - rp}",
    x_min=0, x_max=4, valid_xrange=valid_ranges["bp_rp"],
    ax=axes[0], **common_kwds)

plot_rv_completeness(
    data["phot_rp_mean_mag"], 
    latex_label_name=r"\textrm{apparent rp magnitude}",
    x_min=6, x_max=12.4, valid_xrange=valid_ranges["phot_rp_mean_mag"],
    ax=axes[1], **common_kwds)

for ax in axes:
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylim(-0.05, 1.05)

fig.tight_layout()

fig.savefig("figures/sb2_rvs_completeness.png", dpi=150)
fig.savefig("figures/sb2_rvs_completeness.pdf", dpi=300)


in_sb2_source_parameter_range = np.ones(len(data), dtype=bool)
for label_name, (lower_value, upper_value) in valid_ranges.items():
    if lower_value is not None:
        in_sb2_source_parameter_range *= (data[label_name] >= lower_value)
    
    if upper_value is not None:
        in_sb2_source_parameter_range *= (upper_value >= data[label_name])
        
finite_rv = np.isfinite(data["radial_velocity"])
is_sb2 = in_sb2_source_parameter_range * ~finite_rv


N_in_sb2_sp_range = sum(in_sb2_source_parameter_range)
N_is_sb2 = sum(is_sb2)

print("""
Total sources in valid SB2 range: {0:.0f}
Numer of sources in that range without an RV: {1:.0f}
""".format(N_in_sb2_sp_range, N_is_sb2))



# Plot the source property distributions of the SB2 candidate systems relative to a control sample.

# For each SB2 candidate, we need to find the closest star in (bp - rp, apparent rp mag, absolute rp mag)
absolute_rp_mag = data["phot_rp_mean_mag"] + 5 * np.log10(data["parallax"]/100.0)
X = np.vstack([
    data["bp_rp"],
    data["phot_rp_mean_mag"],
    absolute_rp_mag
]).T

# Build a k-d tree using the stars that are NOT SB2s. 
# We can query this in parameter space that we care about to get the clsoest non-SB2.
import npm_utils as npm
finite = np.all(np.isfinite(X), axis=1)
kdt_indices = np.where(finite * ~is_sb2)[0]

kdt, scale, offset = npm.build_kdtree(X[kdt_indices], 
                                      relative_scales=[0.1, 1.0, 1.0])
kdt_kwds = dict(offset=offset, scale=scale, 
                minimum_points=1, maximum_points=1)

sb2_indices = np.arange(len(data))[is_sb2]
control_indices = np.nan * np.ones_like(sb2_indices)
K = sb2_indices.size

import tqdm
for i, index in tqdm.tqdm(enumerate(sb2_indices), total=K):
    try:
        indices_returned = list(kdt_indices[kdt.query(X[[index]], 1, return_distance=False)][0])
    
    except ValueError:
        continue
    
    control_indices[i] = indices_returned[0]



comp_subset = np.isfinite(control_indices)
subset_sb2_indices = sb2_indices[comp_subset]
subset_cnt_indices = control_indices[comp_subset].astype(int)


astrometric_unit_weight_error = np.sqrt(data["astrometric_chi2_al"]/(data["astrometric_n_obs_al"] - 5))


from mpl_utils import plot_histogram_steps


fig, axes = plt.subplots(3, 1, figsize=(3.4, 10))


B = 50
bins = np.linspace(0.5, 3, B)

sb2_color, control_color = ("tab:red", "#666666")
sb2_label, control_label = (r"\textrm{SB2 candidates}", r"\textrm{control sample}")

fill_alpha = 0.3

def plot_twosamples(ax, bins, control_sample, sb2_sample):

    x = bins[:-1] + 0.5 * np.diff(bins)
    y = np.histogram(control_sample, bins=bins)
    ax.plot(x, y[0], "-",
            c=control_color, drawstyle="steps-mid", label=control_label)
    ax.errorbar(x, y[0], yerr=np.sqrt(y[0]),
                c=control_color, fmt=None, ecolor=control_color)


    xx = np.array(x).repeat(2)[1:]
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1]) - xstep/2.0

    yy = np.array(y[0]).repeat(2)

    ax.fill_between(xx, 0, yy, facecolor=control_color, alpha=fill_alpha)


    y = np.histogram(sb2_sample, bins=bins)
    ax.plot(x, y[0], "-",
            c=sb2_color, drawstyle="steps-mid", label=sb2_label)
    ax.errorbar(x, y[0],
                yerr=np.sqrt(y[0]),
                c=sb2_color, fmt=None, ecolor=sb2_color)

    xx = np.array(x).repeat(2)[1:]
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1]) - xstep/2.0

    yy = np.array(y[0]).repeat(2)

    ax.fill_between(xx, 0, yy, facecolor=sb2_color, alpha=fill_alpha)


plot_twosamples(axes[0], bins, 
                astrometric_unit_weight_error[subset_cnt_indices],
                astrometric_unit_weight_error[subset_sb2_indices])

axes[0].set_xlabel(r"\textrm{astrometric unit weight error}")#$u = (\chi_{al}^2/\nu)^{1/2}$")
axes[0].set_ylabel(r"\textrm{count / $10^{4}$}")

axes[0].legend(frameon=False)

def do_yticks(ax, step=10000):
    ticks = np.arange(0, 10*step, step)
    ax.set_yticks(ticks)
    ax.set_ylim(0, ticks[-1])
    ax.set_yticklabels([r"${:.0f}$".format(ea) for ea in (ticks/10000).astype(int)])
    return ticks
do_yticks(axes[0])
axes[0].xaxis.set_major_locator(MaxNLocator(6))


"""
axes[1].hist(data["phot_bp_rp_excess_factor"][subset_cnt_indices], 
             facecolor=control_color, edgecolor=control_color, label=control_label, **hist_kwds)
axes[1].hist(data["phot_bp_rp_excess_factor"][subset_sb2_indices], 
             facecolor=sb2_color, edgecolor=sb2_color, label=sb2_label, **hist_kwds)
"""

plot_twosamples(axes[1], np.linspace(1.2, 1.5, B), 
                data["phot_bp_rp_excess_factor"][subset_cnt_indices],
                data["phot_bp_rp_excess_factor"][subset_sb2_indices])

axes[1].set_xlabel(r"\textrm{phot bp - rp excess factor}")
axes[1].set_ylabel(r"\textrm{count / $10^4$}")
#axes[1].set_ylim(0, 10000

axes[1].xaxis.set_major_locator(MaxNLocator(4))
axes[1].legend(frameon=False)
do_yticks(axes[1])

phot_g_variability = np.log10(np.sqrt(data["astrometric_n_good_obs_al"]) \
                   * data["phot_g_mean_flux_error"] / data["phot_g_mean_flux"])


plot_twosamples(axes[2], np.linspace(-3, -1, B), 
                phot_g_variability[subset_cnt_indices],
                phot_g_variability[subset_sb2_indices])
axes[2].set_xlabel(r"$\log_{10}\left(\textrm{photometric variability}\right)$")
axes[2].set_ylabel(r"\textrm{count / $10^4$}")
axes[2].xaxis.set_major_locator(MaxNLocator(6))
do_yticks(axes[2])
axes[2].legend(frameon=False)

fig.tight_layout()

offset = 0.05
for axlabel, ax in zip("abc", axes):
    ax.text(offset, 1 - offset, r"\textrm{{({0})}}".format(axlabel), transform=ax.transAxes,
        horizontalalignment="left", verticalalignment="top")

fig.tight_layout()
fig.savefig("figures/sb2_histograms.png", dpi=150)
fig.savefig("figures/sb2_histograms.pdf", dpi=300)




# Plot the completeness fraction across the H-R diagram.
def plot_density_fraction(x, y, N_bins=150, 
                          min_points_per_bin=5, x_min=None, x_max=None, 
                          y_min=None, y_max=None, ax=None, xlabel=None, 
                          ylabel=None, figsize=(8, 8), colorbar=True,
                          log=False,
                          mask=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
        
    finite = np.isfinite(x * y)
    if x_min is not None:
        finite *= (x >= x_min)
    if x_max is not None:
        finite *= (x_max >= x)
    if y_min is not None:
        finite *= (y >= y_min)
    if y_max is not None:
        finite *= (y_max >= y)

    den = finite
    if mask is not None:
        den *= mask
        
    num = (~finite_rv) * den

    H_all, xedges, yedges = np.histogram2d(x[den], y[den],
        bins=N_bins)

    H_bin, _, __ = np.histogram2d(x[num], y[num],
        bins=(xedges, yedges))


    H = H_bin/H_all.astype(float)
    H[H_all < min_points_per_bin] = np.nan
    if log:
        H = np.log(1 + H)

    print(np.nanmin(H), np.nanmax(H))
    kwds = dict(
        aspect=np.ptp(xedges)/np.ptp(yedges), 
        extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
        cmap="inferno",
    )
    kwds.update(kwargs)

    image = ax.imshow(H.T, **kwds)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)  

    if colorbar:
        cax = fig.add_axes([0.90, 0.125, 0.05, 0.75])

        cbar = plt.colorbar(image, cax=cax, orientation="vertical")
        cbar.set_label(r"\textrm{fraction of stars without radial velocity}")
    
    return fig




fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))

B = 100
kwds = dict(x=data["l"], y=data["b"], N_bins=(2 * B, B), aspect=0.5,
            vmin=0, vmax=0.5, colorbar=False, cmap="Greys",
            xlabel=r"$l$\textrm{ / deg}", ylabel=r"$b$\textrm{ / deg}")

plot_density_fraction(ax=axes[0], mask=None, **kwds)
plot_density_fraction(ax=axes[1], mask=in_sb2_source_parameter_range, **kwds)

for ax in axes:
    ax.set_xticks(np.arange(0, 361, 45).astype(int))
fig.tight_layout()


fig.savefig("figures/sb2_sky_structure.png", dpi=150)
fig.savefig("figures/sb2_sky_structure.pdf", dpi=300)
