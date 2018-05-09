
"""
Code for the velociraptor project.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

import stan_utils as stan
import mpl_utils

plt.style.use(mpl_utils.mpl_style)

np.random.seed(42)

def load_gaia_sources(path, N=None, **kwargs):
    """
    Load a subset of Gaia data and calculate additional properties like
    `absolute_g_mag`, `absolute_bp_mag`, `absolute_rp_mag`, and
    `rv_single_epoch_variance`.

    :param path:
        The local path to load the sources from.

    :param N: [optional]
        If not `None`, draw a random `N` sources from the data.
    """

    sources = Table.read(path, **kwargs)

    for band in ("g", "bp", "rp"):
        sources["absolute_{}_mag".format(band)] = \
              sources["phot_{}_mean_mag".format(band)] \
            + 5 * np.log10(sources["parallax"]/100.0)

    sources["rv_single_epoch_variance"] = sources["radial_velocity_error"]**2 \
                                        * sources["rv_nb_transits"] * np.pi/2.0

    if N is not None:
        S = len(sources)
        return sources[np.random.choice(S, size=min(int(N), S), replace=False)]
    return sources


def prepare_model(rv_single_epoch_variance, target=0.01, model_path="model.stan",
    **source_params):
    """
    Compile the Stan model, and prepare the data and initialisation dictionaries
    for optimization and sampling.

    :param rv_single_epoch_variance:
        The variance in single epoch measurements of radial velocity.

    :param target: [optional]
        The target radial velocity variance to use when initialising the model
        coefficients.

    :param model_path: [optional]
        The local path of the Stan model.

    :Keyword Arguments:
        * *source_params* (``dict``) These are passed directly to the
        `_rvf_design_matrix`, so they should include all of the source labels
        needed to construct the design matrix (e.g., `phot_rp_mean_flux`). The
        array length for each source label should match that of
        `rv_single_epoch_variance`.

    :returns:
        The compiled Stan model, the data dictionary, the initialsiation
        dictionary, and a mask corresponding to which `sources` were used to
        construct the data dictionary.
    """

    dm = _rvf_design_matrix(**source_params)
    finite = np.all(np.isfinite(dm), axis=0)

    if not all(finite):
        logging.warn("Design matrix contains {0} (of {1}) non-finite values!"\
                     "Exlcuding them from model fit.".format(
                        sum(~finite), finite.size))
    
    dm = dm[:, finite]
    coeff = _rvf_initial_coefficients(dm, target=target)

    init = dict(theta=0.1, mu_coefficients=coeff, sigma_coefficients=coeff)
    data = dict(N=sum(finite), rv_variance=rv_single_epoch_variance[finite],
        design_matrix=dm.T, M=dm.shape[0])

    model = stan.load_stan_model(model_path)

    return (model, data, init, finite)


def _rvf_design_matrix(phot_rp_mean_flux, bp_rp, **kwargs):
    """
    Design matrix for the radial velocity floor variance.

    # TODO: Should we check for finite-ness here?
    """
    return np.array([
        np.ones(len(phot_rp_mean_flux)),
        phot_rp_mean_flux**-1,
        phot_rp_mean_flux**-2,
        bp_rp**-1,
        bp_rp**-2
    ])



def _rvf_initial_coefficients(design_matrix, target=0.01):
    """
    Initial coefficients for the model.
    """
    return target / (design_matrix.shape[0] * np.nanmean(design_matrix, axis=1))


def predict_map_rv_single_epoch_variance(samples, **source_params):
    """
    Predict the maximum a-posteriori estimate of the radial velocity variance
    from a single epoch.
    
    :param samples:
        The Stan chains from the model.
    """

    params = samples.extract(("mu_coefficients", "sigma_coefficients"))

    dm = _rvf_design_matrix(**source_params)
    mu = np.dot(np.mean(params["mu_coefficients"], axis=0), dm)
    sigma = np.dot(np.mean(params["sigma_coefficients"], axis=0), dm)
    
    return (mu, sigma)


def binary_probability(samples, **source_params):
    """
    Calculate the probability of binarity for Gaia sources, based on samples
    from our MCMC.
    """

    max_rv_variance = np.nanmax(source_params["rv_single_epoch_variance"])


    log_ps1 = np.log(samples["theta"]) - np.log(max_rv_variance)
    log_ps2 = np.log(1 - samples["theta"]) 

    raise NotImplementedError()



def plot_model_predictions_corner(samples, sources=None, parameter_limits=None,
    log_parameters=None, N=100, labels=None, **kwargs):
    """
    Make a corner plot showing the maximum a posteori radial velocity variance
    for different (stellar) properties that contribute to the model.
    
    :param samples:
        The MCMC samples.

    :param sources: [optional]
        A table of Gaia sources. This is used to determine the bounds on the
        parameters. If `None` is given, then `parameter_limits` should be
        given instead. If `sources` and `parameter_limits` are given, then the
        `parameter_limits` will supercede those calculated from `sources.

    :param parameter_limits: [optional]
        A dictionary containing source parameters as keys and a two-length
        tuple containing the lower and upper bounds of the parameter.

    :param log_parameters: [optional]
        A tuple containing the parameter names that should be shown in log space

    ""
    """

    parameter_names = tuple(
        set(_rvf_design_matrix.__code__.co_varnames).difference(["kwargs"]))

    limits = dict()
    if sources is not None:
        for pn in parameter_names:
            limits[pn] = (np.nanmin(sources[pn]), np.nanmax(sources[pn]))
    else:
        missing = tuple(set(parameter_names).difference(parameter_limits))
        if len(missing) > 0:
            raise ValueError("missing parameter limits for {}".format(
                ", ".join(missing)))

    if parameter_limits is not None:
        limits.update(parameter_limits)

    if log_parameters is None:
        log_parameters = []

    if labels is None:
        labels = dict()

    def mesh(parameter_name):
        v = limits[parameter_name]
        s, e = (np.min(v), np.max(v))

        if parameter_name not in log_parameters:
            return np.linspace(s, e, N)
        else:
            return np.logspace(np.log10(s), np.log10(e), N)

    samples_kwd = kwargs.get("samples_kwd", "mu_coefficients")
    coefficients = np.mean(samples.extract((samples_kwd, ))[samples_kwd], axis=0)
    
    P = len(limits)

    # Calculate the expected radial velocity variance for all combinations of
    # parameters.
    combinations = np.meshgrid(*[mesh(pn) for pn in parameter_names])
    grid_combinations = np.vstack([comb.flatten() for comb in combinations])

    expectation = np.dot(coefficients, _rvf_design_matrix(**dict(zip(
        parameter_names, grid_combinations))))

    fig, axes = plt.subplots(P, P, figsize=(6 * P, 6 * P))
    axes = np.atleast_2d(axes)


    for i, x_param in enumerate(parameter_names):
        for j, y_param in enumerate(parameter_names):

            ax = axes[j, i]

            if i > j:
                ax.set_visible(False)
                continue

            elif i == j:
                x = grid_combinations[i]

                # Get the mean at each unique x.
                x_uniques = np.sort(np.unique(x))
                y_percentiles = np.zeros((3, x_uniques.size), dtype=float)

                for k, x_unique in enumerate(x_uniques):
                    match = (grid_combinations[i] == x_unique)
                    y_percentiles[:, k] = np.percentile(
                        expectation[match], [0, 50, 100])

                ax.plot(x_uniques, y_percentiles[1], "r-")
                ax.fill_between(
                    x_uniques, y_percentiles[0], y_percentiles[2],
                    facecolor="r", alpha=0.3, edgecolor="none")
                
                if x_param in log_parameters:
                    ax.semilogx()

                ax.set_xlabel(labels.get(x_param, x_param))
                ax.set_ylabel(r"\textrm{single epoch radial velocity variance}"\
                              r" $(\textrm{km}^2\,\textrm{s}^{-2})$")
            
            else:
                x, y = grid_combinations[[i, j]]

                #_x = np.log10(x) if x_param in log_parameters else x
                #_y = np.log10(y) if y_param in log_parameters else y
                
                imshow_kwds = dict(cmap="Reds", aspect="equal",
                    extent=(np.min(x), np.max(x), np.max(y), np.min(y)))
                
                if x_param in log_parameters:
                    ax.semilogx()

                if y_param in log_parameters:
                    ax.semilogy()

                ax.imshow(expectation.reshape((N, N)), **imshow_kwds)
                
                ax.set_xlabel(labels.get(x_param, x_param))
                ax.set_ylabel(labels.get(y_param, y_param))

    fig.tight_layout()

    return fig
