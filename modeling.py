
"""
Modeling and validation of the radial velocity calibration model.
"""

import numpy as np
import matplotlib.pyplot as plt

import velociraptor

# Load the data and make some plots.
sources = velociraptor.load_gaia_sources("data/rv-cal-subset.fits.gz", N=1e4)

model, data_dict, init_dict, used_in_fit = velociraptor.prepare_model(**sources)

p_opt = model.optimizing(data=data_dict, init=init_dict)

samples = model.sampling(**velociraptor.stan.sampling_kwds(
    data=data_dict, chains=2, iter=2000, init=p_opt))


fig = velociraptor.plot_model_predictions_corner(samples, sources,
    log_parameters=("phot_rp_mean_flux", ), labels=dict(
        phot_rp_mean_flux=r"\textrm{mean rp flux}",
        bp_rp=r"\textrm{bp-rp}"))
fig.savefig("figures/model-predictions.pdf", dpi=150)

scatter_kwds = dict(s=1, facecolor="#000000", alpha=0.5, rasterized=True)
model_scatter_kwds = dict(s=1, facecolor="r", alpha=0.5, rasterized=True)


model_rv_sev_mu, model_rv_sev_sigma = \
    velociraptor.predict_map_rv_single_epoch_variance(samples, **sources)

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
prob_binarity = np.mean(np.exp(samples["log_membership_probability"]), axis=0)
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


qc_giants = qc * (3 < sources["bp_rp"]) * (sources["bp_rp"] < 1.0) \
          * (sources["absolute_g_mag"] > 2.5)
qc_dwarfs = qc * (3 < sources["bp_rp"]) * (sources["bp_rp"] < 1.0) \
          * (sources["absolute_g_mag"] < 2.5)




