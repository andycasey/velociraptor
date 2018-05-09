
"""
Code for the velociraptor project.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import stan_utils as stan
from mpl_style import mpl_style

plt.style.use(mpl_style)

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
        return sources[np.random.choice(S, size=min(N, S), replace=False)]
    return sources


def prepare_model(phot_rp_mean_flux, rv_single_epoch_variance, 
    model_path="model.stan"):
    """
    Compile the Stan model, and prepare the data and initialisation dictionaries
    for optimization and sampling.

    :param phot_rp_mean_flux:
        The mean RP flux.

    :param rv_single_epoch_variance:
        The variance in single epoch measurements of radial velocity.

    :param model_path: [optional]
        The local path of the Stan model.

    :returns:
        The compiled Stan model, the data dictionary, the initialsiation
        dictionary, and a mask corresponding to which `sources` were used to
        construct the data dictionary.
    """

    dm = _rvf_design_matrix(phot_rp_mean_flux)
    finite = np.all(np.isfinite(dm), axis=0)

    if not all(finite):
        print("Design matrix contains non-finite values! Masking out.")

    dm = dm[:, finite]
    coeff = _rvf_initial_coefficients(dm)

    init = dict(theta=0.1, mu_coefficients=coeff, sigma_coefficients=coeff)
    data = dict(N=sum(finite), rv_variance=rv_single_epoch_variance,
        design_matrix=dm.T, M=dm.shape[0])

    model = stan.load_stan_model(model_path)

    return (model, data, init, finite)


def _rvf_design_matrix(phot_rp_mean_flux):
    """
    Design matrix for the radial velocity floor variance.
    """
    return np.array([
        np.ones(len(phot_rp_mean_flux)),
        phot_rp_mean_flux**-1,
        phot_rp_mean_flux**-2
    ])


def _rvf_initial_coefficients(design_matrix, target=0.01):
    """
    Initial coefficients for the model.
    """
    return target / (design_matrix.shape[0] * np.nanmean(design_matrix, axis=1))


def predict_map_rv_single_epoch_variance(samples, *params):
    """
    Predict the maximum a-posteriori estimate of the radial velocity variance
    from a single epoch.
    
    :param samples:
        The Stan chains from the model.
    """

    params = samples.extract(("mu_coefficients", "sigma_coefficients"))

    dm = _rvf_design_matrix(*params)
    mu = np.dot(np.mean(params["mu_coefficients"], axis=0), dm)
    sigma = np.dot(np.mean(params["sigma_coefficients"], axis=0), dm)
    
    return (mu, sigma)