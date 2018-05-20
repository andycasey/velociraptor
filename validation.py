
"""
Validation of the radial velocity calibration model.
"""

import numpy as np
import matplotlib.pyplot as plt

import velociraptor

# Load the data and make some plots.
sources = velociraptor.load_gaia_sources("data/rv-cal-subset.fits.gz")

model, data_dict, init_dict, used_in_fit = velociraptor.prepare_model(
    S=1e4, **sources)

init_dict = dict([
    ('theta', 0.15167079461165178),
    #('mu_coefficient', 0.3),
    #('mu_coefficients', np.array([1e12, 1])),
    ('mu_coeff_0', 0.3),
    ('mu_coeff_1', 1e-4)
    ('mu_coeff_2', 1e12),
    ('mu_coeff_3', 1),
    ('mu_coeff_4', 1)
    #('sigma_coefficients', np.array([0.3, 4e11, 1]))
    ('sigma_coeff_0', 0.3),
    ('sigma_coeff_1', 1e-4),
    ('sigma_coeff_2', 4e11),
    ('sigma_coeff_3', 1),
    ('sigma_coeff_4', 1),
])


p_opt = model.optimizing(data=data_dict, init=init_dict)

print(p_opt)

fig, ax = plt.subplots()
ax.scatter(sources["phot_rp_mean_flux"][used_in_fit], data_dict["rv_variance"],
    s=1, alpha=0.5, facecolor="k")

x = np.logspace(
    np.log10(np.nanmin(sources["phot_rp_mean_flux"])),
    np.log10(np.nanmax(sources["phot_rp_mean_flux"])),
    1000)
bp_rp = np.nanmean(sources["bp_rp"]) * np.ones(x.size)
mu_opt = np.dot(
    np.hstack([p_opt["mu_coeff_0"], p_opt["mu_coeff_1"], p_opt["mu_coeff_2"], p_opt["mu_coeff_3"], p_opt["mu_coeff_4"]]),
    velociraptor._rvf_design_matrix(x, bp_rp=bp_rp))
sigma_opt = np.dot(
    p_opt["sigma_coefficients"],
    velociraptor._rvf_design_matrix(x, bp_rp=bp_rp))


mu_init = np.dot(
    np.hstack([init_dict["mu_coeff_0"], init_dict["mu_coeff_1"], init_dict["mu_coeff_2"], p_opt["mu_coeff_3"], p_opt["mu_coeff_4"]]),
    velociraptor._rvf_design_matrix(x, bp_rp=bp_rp))


sigma_init = np.dot(
    np.hstack([init_dict["sigma_coeff_0"], init_dict["sigma_coeff_1"], init_dict["sigma_coeff_2"], p_opt["sigma_coeff_3"], p_opt["sigma_coeff_4"]]),
    velociraptor._rvf_design_matrix(x, bp_rp=bp_rp))



ax.plot(x, mu_init, c='r')
ax.fill_between(x, mu_init - sigma_init, mu_init + sigma_init, facecolor="r",
    alpha=0.3, edgecolor="none")

ax.plot(x, mu_opt, c='b')
ax.fill_between(x, mu_opt - sigma_opt, mu_opt + sigma_opt, facecolor="b",
    alpha=0.3, edgecolor="none")

ax.semilogx()
ax.set_ylim(0, 400)


raise a


iterations = 2000
p_samples = model.sampling(**velociraptor.stan.sampling_kwds(
    data=data_dict, chains=2, iter=iterations, init=init_dict))

print(p_samples)

parameter_names = ("theta", "mu_coeff_0",  "mu_coeff_1", "mu_coeff_2",
    "mu_coeff_3", "mu_coeff_4", "sigma_coeff_0", "sigma_coeff_1", 
    "sigma_coeff_2", "sigma_coeff_3", "sigma_coeff_4")

chains = p_samples.extract(parameter_names, permuted=True)
chains = np.vstack(
    [chains[pn].reshape((-1, iterations)) for pn in parameter_names]).T

initial = np.hstack([init_dict[pn] for pn in parameter_names])

import corner
fig = corner.corner(chains, truths=initial)

# Make many draws.

x = np.logspace(
    np.log10(np.nanmin(sources["phot_rp_mean_flux"])),
    np.log10(np.nanmax(sources["phot_rp_mean_flux"])),
    1000)
bp_rp = np.nanmean(sources["bp_rp"]) * np.ones(x.size)
dm = velociraptor._rvf_design_matrix(x, bp_rp=bp_rp)

fig, ax = plt.subplots()
ax.scatter(sources["phot_rp_mean_flux"][used_in_fit], data_dict["rv_variance"],
    s=1, alpha=0.5, facecolor="#000000")


for index in np.random.choice(2000, 100, replace=False):
    y = np.dot(chains[index][1:4], dm) \
      + np.random.normal(0, 1) * np.dot(chains[index][4:], dm)

    ax.plot(x, y, "r-", alpha=0.05)

ax.semilogx()
ax.set_ylim(0, ax.get_ylim()[1])



"""
fig = velociraptor.plot_model_predictions_corner(p_samples, sources,
    log_parameters=("phot_rp_mean_flux", ), labels=dict(
        phot_rp_mean_flux=r"\textrm{mean rp flux}",
        bp_rp=r"\textrm{bp-rp}"))
fig.savefig("figures/model-predictions.pdf", dpi=150)

scatter_kwds = dict(s=1, facecolor="#000000", alpha=0.5, rasterized=True)
model_scatter_kwds = dict(s=1, facecolor="r", alpha=0.5, rasterized=True)


model_rv_sev_mu, model_rv_sev_sigma = \
    velociraptor.predict_map_rv_single_epoch_variance(p_samples, **sources)
"""

# Make a corner plot from our p_samples.


N_mu = 3 # p_samples.extract("mu_coefficients")["mu_coefficients"].shape[1]
N_sigma = 3# p_samples.extract("sigma_coefficients")["sigma_coefficients"].shape[1]



raise a

y = np.dot(
    np.median(chains, axis=0)[1:1 + N_mu], 
    )
ax.plot(x, y, c='r')
raise a




labels = (
    r"$\theta$",
    r"$\mu_0$",
    r"$\mu_1$",
    r"$\mu_2$",
    r"$\mu_3$",
    r"$\mu_4$",
    r"$\mu_5$",
    r"$\sigma_0$",
    r"$\sigma_1$",
    r"$\sigma_2$",
    r"$\sigma_3$",
    r"$\sigma_4$",
    r"$\sigma_5$",
)
import corner

fig = corner.corner(chains, labels=labels, truths=initial)

# Calculate SB2 probability distributions for all stars.
masks = [
    np.ones(len(chains), dtype=bool)
]
"""
    (chains.T[1] < 10),
    ((25 >= chains.T[1]) * (chains.T[1] > 15)),
    ((35 >= chains.T[1]) * (chains.T[1] > 25)),
    (chains.T[1] > 40)
]
"""

x = np.logspace(
    np.log10(np.nanmin(sources["phot_rp_mean_flux"])),
    np.log10(np.nanmax(sources["phot_rp_mean_flux"])),
    1000)

from matplotlib.cm import Set1
bp_rp = np.nanmean(sources["bp_rp"]) * np.ones(x.size)

fig, ax = plt.subplots()
ax.semilogx()

for i, mask in enumerate(masks):

    y = np.dot(chains[:, 1:4], velociraptor._rvf_design_matrix(x, bp_rp=bp_rp))
    q = np.percentile(y, [16, 50, 84], axis=0)

    c = Set1(i)

    ax.plot(x, q[1], label="set {}".format(i),
        alpha=0.5, c=c, lw=2)
    ax.fill_between(x, q[0], q[2], facecolor=c, alpha=0.3, edgecolor="none")




raise a

# Plot the model against various properties of interest.
shorthand_parameters = [
    ("phot_rp_mean_flux", "mean rp flux", True),
    ("phot_bp_mean_flux", "mean bp flux", True),
    ("phot_g_mean_flux", "mean g flux", True),
    ("bp_rp", "bp-rp colour", False),
    ("phot_g_mean_mag", "mean g magnitude", False),
    ("phot_rp_mean_mag", "mean rp magnitude", False),
    ("phot_bp_mean_mag", "mean bp magnitude", False),
    ("teff_val", "inferred temperature", False),
    ("ra", "right ascension", False),
    ("dec", "declination", False),
    ("radial_velocity", "radial velocity", False),
    ("rv_nb_transits", "number of radial velocity transits", False),
    ("absolute_g_mag", "absolute g magnitude", False),
    ("absolute_bp_mag", "absolute bp magnitude", False),
    ("absolute_rp_mag", "absolute rp magnitude", False),
    ("rv_template_teff", "rv template teff", False),
    ("rv_template_logg", "rv template logg", False),
    ("rv_template_fe_h", "rv template [Fe/H]", False),
]

for label_name, description, is_semilogx in shorthand_parameters:

    x = sources[label_name]
    y = sources["rv_single_epoch_variance"] - model_rv_sev_mu

    idx = np.argsort(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axes:
        ax.scatter(x, y, **model_scatter_kwds)

        if is_semilogx:
            ax.semilogx()

        ax.set_xlabel(r"\textrm{{{0}}}".format(description))
        ax.set_ylabel(r"\textrm{residual single epoch radial velocity variance} $(\textrm{km}^2\,\textrm{s}^{-2})$")
        

    axes[0].set_title(r"\textrm{all points}")
    axes[1].set_title(r"\textrm{5th-95th percentile in} $y$\textrm{-axis}")

    axes[1].set_ylim(*np.nanpercentile(y, [5, 95]))

    fig.tight_layout()


# Calculate excess given number of transits.
# TODO: Do this somewhere else?
scalar = 2.0 / (np.pi * sources["rv_nb_transits"])
model_rv_error_mu = np.sqrt(model_rv_sev_mu * scalar)
model_rv_error_sigma_pos = \
    np.sqrt((model_rv_sev_mu + model_rv_sev_sigma) * scalar) - model_rv_error_mu
model_rv_error_sigma_neg = \
    np.sqrt((model_rv_sev_mu - model_rv_sev_sigma) * scalar) - model_rv_error_mu 


x = sources["phot_rp_mean_flux"]
y = sources["rv_single_epoch_variance"]
idx = np.argsort(x)

fig, ax = plt.subplots()
ax.scatter(x, y, **scatter_kwds)
ax.plot(x[idx], model_rv_sev_mu[idx], "r-")
ax.fill_between(
    x[idx], 
    (model_rv_sev_mu - model_rv_sev_sigma)[idx], 
    (model_rv_sev_mu + model_rv_sev_sigma)[idx], 
    facecolor="r", alpha=0.3, edgecolor="none")
ax.set_xlim(np.nanmin(x), np.nanmax(x))
ax.set_ylim(-0.5, 2 * np.nanmax(model_rv_sev_mu + model_rv_sev_sigma))
ax.semilogx()
ax.set_xlabel(r"\textrm{rp mean flux}")
ax.set_ylabel(r"\textrm{single epoch radial velocity variance} $(\textrm{km}^2\,\textrm{s}^{-2})$")
fig.tight_layout()


x = sources["phot_rp_mean_flux"]
y = sources["radial_velocity_error"]
idx = np.argsort(x)

fig, ax = plt.subplots()
ax.scatter(x, y, **scatter_kwds)
ax.plot(x[idx], model_rv_error_mu[idx], "r-")
ax.fill_between(
    x[idx], 
    (model_rv_error_mu + model_rv_error_sigma_pos)[idx],
    (model_rv_error_mu + model_rv_error_sigma_neg)[idx],
    facecolor="r", alpha=0.3, edgecolor="none")
ax.set_xlim(np.nanmin(x), np.nanmax(x))
ax.set_ylim(-0.5, 2 * np.nanmax(model_rv_error_mu + model_rv_error_sigma_pos))
ax.semilogx()
ax.set_xlabel(r"\textrm{rp mean flux}")
ax.set_ylabel(r"\textrm{radial velocity error} $(\textrm{km\,s}^{-1})$")
fig.tight_layout()



# Plot the model residuals against different stellar properties.
y = sources["rv_single_epoch_variance"]
mu = model_rv_sev_mu
sigma = model_rv_sev_sigma

for label_name, description, is_semilogx in shorthand_parameters:

    fig, ax = plt.subplots()

    x = sources[label_name]
    idx = np.argsort(x)

    ax.scatter(x, y - mu, **scatter_kwds)
    ax.axhline(0, c="r")
    ax.fill_between(x[idx], -sigma[idx], +sigma[idx], facecolor="r", alpha=0.3,
        edgecolor="none")
    ax.set_xlim(np.nanmin(x), np.nanmax(x))

    # TODO: set ylim
    abs_ylim = 2 * np.nanmax(sigma)
    ax.set_ylim(-abs_ylim, +abs_ylim)

    if is_semilogx:
        ax.semilogx()

    ax.set_xlabel(r"\textrm{{{0}}}".format(description))
    ax.set_ylabel(r"\textrm{residual single epoch radial valocity variance} $(\textrm{km}^2\,\textrm{s}^{-2})$")

    fig.tight_layout()


# Plot the binary fraction with different parameters.
N_bins = 15
prob_binarity = np.mean(np.exp(p_samples["log_membership_probability"]), axis=0)
for label_name, description, is_semilogx in shorthand_parameters:

    x = sources[label_name][used_in_fit]
    y = prob_binarity

    finite = np.isfinite(x * y)
    x, y = (x[finite], y[finite])

    if is_semilogx:
        es_bins = np.logspace(
            np.log10(x.min()), np.log10(x.max()), N_bins)
    else:
        es_bins = np.linspace(x.min(), x.max(), N_bins)

    es_bin_centers = es_bins[:-1] + np.diff(es_bins)/2.
    es_binarity = np.zeros(es_bins.size - 1, dtype=float)
    es_binarity_error = np.zeros_like(es_binarity)

    for i, left_edge in enumerate(es_bins[:-1]):
        right_edge = es_bins[i + 1]

        in_bin = (right_edge > x) * (x >= left_edge)
        es_binarity[i] += np.sum((y > 0.5) * in_bin) \
                        / np.sum(in_bin)
        es_binarity_error[i] += np.sqrt(np.sum((y > 0.5) * in_bin)) \
                              / np.sum(in_bin)

    if is_semilogx:
        ed_bins = 10**np.percentile(np.log10(x), np.linspace(0, 100, N_bins))
    else:
        ed_bins = np.percentile(x, np.linspace(0, 100, N_bins))

    ed_bin_centers = ed_bins[:-1] + np.diff(ed_bins)/2.
    ed_binarity = np.zeros(ed_bins.size - 1, dtype=float)
    ed_binarity_error = np.zeros_like(ed_binarity)

    for i, left_edge in enumerate(ed_bins[:-1]):
        right_edge = ed_bins[i + 1]

        in_bin = (right_edge > x) * (x >= left_edge)
        ed_binarity[i] += np.sum((y > 0.5) * in_bin) \
                        / np.sum(in_bin)
        ed_binarity_error[i] += np.sqrt(np.sum((y > 0.5) * in_bin))\
                              / np.sum(in_bin)


    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    xx, yy = velociraptor.mpl_utils.plot_histogram_steps(
        axes[0], es_bin_centers, es_binarity, es_binarity_error)
    axes[0].set_title(r"\textrm{equi-spaced bins}")

    xx, yy = velociraptor.mpl_utils.plot_histogram_steps(
        axes[1], ed_bin_centers, ed_binarity, ed_binarity_error)
    axes[1].set_title(r"\textrm{equi-density bins}")
    
    ylims = np.hstack([ax.get_ylim() for ax in axes])
    for ax in axes:
        ax.set_xlim(xx[0], xx[-1])
        ax.set_ylim(np.min(ylims), np.max(ylims))
        if is_semilogx: ax.semilogx()

        ax.set_ylabel(r"\textrm{binary fraction}")
        ax.set_xlabel(r"\textrm{{{0}}}".format(description))

    fig.tight_layout()


# OK let's do this for a sample of giants and a sample of dwarfs and then
# compare.

qc = (sources["parallax"] > 0) \
   * ((sources["parallax"]/sources["parallax_error"]) > 5)

fig, ax = plt.subplots()
ax.scatter(sources["bp_rp"], sources["absolute_g_mag"], **scatter_kwds)
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xlabel(r"\textrm{bp - rp}")
ax.set_ylabel(r"\textrm{absolute g magnitude}")
fig.tight_layout()


qc_giants = qc * (3 > sources["bp_rp"]) * (sources["bp_rp"] > 1.0) \
          * (sources["absolute_g_mag"] < 3)
qc_dwarfs = qc * (3 > sources["bp_rp"]) * (sources["bp_rp"] > 1.0) \
          * (sources["absolute_g_mag"] > 4)


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax in axes:
    ax.scatter(
        sources["bp_rp"][qc],
        sources["absolute_g_mag"][qc],
        **scatter_kwds)

    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xlabel(r"\textrm{bp - rp}")
    ax.set_ylabel(r"\textrm{absolute g magnitude}")

axes[0].scatter(
    sources["bp_rp"][qc_dwarfs],
    sources["absolute_g_mag"][qc_dwarfs],
    s=5, zorder=10)
axes[1].scatter(
    sources["bp_rp"][qc_giants],
    sources["absolute_g_mag"][qc_giants],
    s=5, zorder=10)

axes[0].set_title(r"\textrm{main-sequence stars}")
axes[1].set_title(r"\textrm{giant stars}")

fig.tight_layout()


print("Number of stars in dwarf model: {}".format(sum(qc_dwarfs)))
print("Number of stars in giant model: {}".format(sum(qc_giants)))



# Run the model for each subset.
dwarf_model, dwarf_data_dict, dwarf_init_dict, dwarf_used_in_fit \
    = velociraptor.prepare_model(**sources[qc_dwarfs])

dwarf_p_opt = dwarf_model.optimizing(data=dwarf_data_dict, init=dwarf_init_dict)
dwarf_samples = dwarf_model.sampling(**velociraptor.stan.sampling_kwds(
    data=dwarf_data_dict, chains=2, iter=2000, init=dwarf_p_opt))



giant_model, giant_data_dict, giant_init_dict, giant_used_in_fit \
    = velociraptor.prepare_model(**sources[qc_giants])

giant_p_opt = giant_model.optimizing(data=giant_data_dict, init=giant_init_dict)
giant_samples = giant_model.sampling(**velociraptor.stan.sampling_kwds(
    data=giant_data_dict, chains=2, iter=2000, init=giant_p_opt))




# Plot the performance of the two models on the same figure.
giant_model_rv_sev_mu, giant_model_rv_sev_sigma = \
    velociraptor.predict_map_rv_single_epoch_variance(giant_samples, **sources[qc_giants])

dwarf_model_rv_sev_mu, dwarf_model_rv_sev_sigma = \
    velociraptor.predict_map_rv_single_epoch_variance(dwarf_samples, **sources[qc_dwarfs])

shorthand_parameters = [
    ("phot_rp_mean_flux", "mean rp flux", True, True),
    ("phot_bp_mean_flux", "mean bp flux", True, True),
    ("phot_g_mean_flux", "mean g flux", True, True),
    ("bp_rp", "bp-rp colour", False, True),
    ("phot_g_mean_mag", "mean g magnitude", False, True),
    ("phot_rp_mean_mag", "mean rp magnitude", False, True),
    ("phot_bp_mean_mag", "mean bp magnitude", False, True),
    ("teff_val", "inferred temperature", False, True),
    ("rv_template_teff", "rv template teff", False, False),
    ("rv_template_logg", "rv template logg", False, False),
    ("rv_template_fe_h", "rv template [Fe/H]", False, False),
]

def _show_text_upper_right(ax, text):
    return ax.text(0.95, 0.95, text, transform=ax.transAxes,
        horizontalalignment="right", verticalalignment="top")

def _show_number_of_data_points(ax, N):
    return _show_text_upper_right(ax, r"${0:.0f}$".format(N))


def _smooth_model(x, y, yerr, is_semilogx, average_function=np.mean, 
    equidensity=True, N_smooth_points=30):

    N = min(N_smooth_points, len(set(x)))
    if equidensity:
        if is_semilogx:
            xb = 10**np.percentile(np.log10(x), np.linspace(0, 100, N))
        else:
            xb = np.percentile(x, np.linspace(0, 100, N))

    else:
        if is_semilogx:
            xb = np.logspace(np.log10(x.min()), np.log10(x.max()), N)
        else:
            xb = np.linspace(x.min(), x.max(), N)

    xi = xb[:-1] + np.diff(xb)/2.
    yi = np.zeros_like(xi)
    yerri = np.zeros_like(xi)
    for i, left_edge in enumerate(xb[:-1]):
        right_edge = xb[i + 1]
        in_bin = (right_edge > x) * (x >= left_edge)
        yi[i] = average_function(y[in_bin])
        yerri[i] = average_function(yerr[in_bin])

    return (xi, yi, yerri)



for label_name, description, is_semilogx, equidensity in shorthand_parameters:

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = sources[label_name][qc_dwarfs][dwarf_used_in_fit]
    y = dwarf_model_rv_sev_mu
    idx = np.argsort(x)

    axes[0].scatter(
        x,
        sources["rv_single_epoch_variance"][qc_dwarfs][dwarf_used_in_fit],
        facecolor="r", s=1, alpha=0.05, rasterized=True)
    _show_number_of_data_points(axes[0], len(x))

    # Smooth out the effects.
    xi, yi, yerri = _smooth_model(
        x, dwarf_model_rv_sev_mu, dwarf_model_rv_sev_sigma, is_semilogx,
        equidensity=equidensity)


    for ax in (axes[0], axes[2]):
        ax.plot(xi, yi, "r-")
        ax.fill_between(xi, yi - yerri, yi + yerri,
            facecolor="r", alpha=0.3, edgecolor="none")

    x = sources[label_name][qc_giants][giant_used_in_fit]
    y = giant_model_rv_sev_mu
    idx = np.argsort(x)

    axes[1].scatter(
        x,
        sources["rv_single_epoch_variance"][qc_giants][giant_used_in_fit],
        facecolor="b", s=1, alpha=0.05, rasterized=True)
    _show_number_of_data_points(axes[1], len(x))
    
    xi, yi, yerri = _smooth_model(
        x, giant_model_rv_sev_mu, giant_model_rv_sev_sigma, is_semilogx,
        equidensity=equidensity)

    for ax in (axes[1], axes[2]):
        ax.plot(xi, yi, "b-")
        ax.fill_between(
            xi, yi - yerri, yi + yerri,
            facecolor="b", alpha=0.3, edgecolor="none")

    axes[0].set_title(r"\textrm{main-sequence stars}")
    axes[1].set_title(r"\textrm{giant stars}")

    x = sources[label_name][qc]
    xlims = (np.nanmin(x), np.nanmax(x))
    for ax in axes:
        ax.set_xlabel(r"\textrm{{{0}}}".format(description))
        ax.set_ylabel(r"\textrm{single epoch radial velocity variance} $(\textrm{km}^2\,\textrm{s}^{-2})$")
        
        if is_semilogx:
            ax.semilogx()

        ax.set_xlim(xlims)
        ax.set_ylim(axes[2].get_ylim())

    _show_text_upper_right(
        axes[2], 
        r"$N_\textrm{{model bins}} = {0}$ \textrm{{(equi-{1})}}".format(
            len(xi) + 1, "density" if equidensity else "spaced"))

    fig.tight_layout()

    fig.savefig(
        "figures/giant-vs-dwarf-{}.pdf".format(label_name.replace("_", "-")),
        dpi=150)



