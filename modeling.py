
"""
Modeling and validation of the radial velocity calibration model.
"""

import numpy as np
import matplotlib.pyplot as plt

import velociraptor

# Load the data and make some plots.
sources = velociraptor.load_gaia_sources("data/rv-cal-subset.fits.gz", N=1e3)

model, data_dict, init_dict, used_in_fit = velociraptor.prepare_model(**sources)

samples = model.sampling(**velociraptor.stan.sampling_kwds(
    data=data_dict, chains=2, iter=2000, init=init_dict))


velociraptor.plot_model_predictions_corner(samples, sources,
    log_parameters=("phot_rp_mean_flux", ), labels=dict(
        phot_rp_mean_flux=r"\textrm{mean rp flux}",
        bp_rp=r"\textrm{bp-rp}"))

raise a



scatter_kwds = dict(s=1, facecolor="#000000", alpha=0.5, rasterized=True)
model_scatter_kwds = dict(s=1, facecolor="r", alpha=0.5, rasterized=True)


model_rv_sev_mu, model_rv_sev_sigma = \
    velociraptor.predict_map_rv_single_epoch_variance(
        samples, phot_rp_mean_flux=sources["phot_rp_mean_flux"])

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
]

for label_name, description, is_semilogx in shorthand_parameters:

    x = sources[label_name]
    y = sources["rv_single_epoch_variance"] - model_rv_sev_mu

    idx = np.argsort(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    """
    ax.plot(x[idx], model_rv_sev_mu[idx], "r-")
    ax.fill_between(
        x[idx],
        (model_rv_sev_mu - model_rv_sev_sigma)[idx],
        (model_rv_sev_mu + model_rv_sev_sigma)[idx],
        facecolor="r", alpha=0.3, edgecolor="none")
    """


    """
    ax.errorbar(x, model_rv_sev_mu, yerr=model_rv_sev_sigma, fmt='.',
        c="r", capsize=0, alpha=0.05)
    """
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

raise a



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



