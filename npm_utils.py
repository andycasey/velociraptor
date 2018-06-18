
import numpy as np
from scipy import (optimize as op, stats)
from sklearn import neighbors as neighbours

import stan_utils as stan


def build_kdt(X_norm, **kwargs):

    kdt_kwds = dict(leaf_size=40, metric="minkowski")
    kdt_kwds.update(kwargs)
    kdt = neighbours.KDTree(X_norm, **kdt_kwds)

    return kdt


def get_ball_around_point(kdt, point, K=1000, full_output=False):
    dist, k_indices = kdt.query(point, K)
    dist, k_indices = (dist[0], k_indices[0])
    return (k_indices, dist) if full_output else k_indices


# Stan needs a finite value to initialize correctly, so we will use a dumb (more
# robust) optimizer to get an initialization value.
def norm_pdf(x, norm_mu, norm_sigma, theta):
    return theta * (2 * np.pi * norm_sigma**2)**(-0.5) * np.exp(-(x - norm_mu)**2/(2*norm_sigma**2))
    

def lognorm_pdf(x, lognorm_mu, lognorm_sigma, theta):
    return (1.0 - theta)/(x * lognorm_sigma * np.sqrt(2*np.pi)) \
           * np.exp(-0.5 * ((np.log(x) - lognorm_mu)/lognorm_sigma)**2)


def ln_likelihood(y, theta, s_mu, s_sigma, b_mu, b_sigma):
    
    s_ivar = s_sigma**-2
    b_ivar = b_sigma**-2
    hl2p = 0.5 * np.log(2*np.pi)
    
    s_lpdf = np.log(theta) - hl2p + 0.5 * np.log(s_ivar) \
           - 0.5 * (y - s_mu)**2 * s_ivar
    
    b_lpdf = np.log(1 - theta) - np.log(y*b_sigma) - hl2p \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar
    ll = np.sum(s_lpdf) + np.sum(b_lpdf)
    #print(lpdf)
    
    assert np.isfinite(ll)
    return ll


def ln_prior(theta, s_mu, s_sigma, b_mu, b_sigma):

    # Ensure that the *mode* of the log-normal distribution is larger than the
    # mean of the normal distribution
    min_mu_multiple = np.log(s_mu) + b_sigma**2

    if not (1 >= theta >= 0) \
    or np.any(s_mu <= 0) \
    or np.any(s_sigma <= 0) \
    or np.any(b_sigma <= 0) \
    or np.any(b_mu < min_mu_multiple):
        return -np.inf

    # Beta prior on theta.
    return stats.beta.logpdf(theta, 5, 5)


def ln_prob(y, L, *params):
    theta, s_mu, s_sigma, b_mu, b_sigma = _unpack_params(params, L=L)
    lp = ln_prior(theta, s_mu, s_sigma, b_mu, b_sigma)
    if np.isfinite(lp):
        return lp + ln_likelihood(y, theta, s_mu, s_sigma, b_mu, b_sigma)
    return lp


def _unpack_params(params, L=None):
    # unpack the multdimensional values.
    if L is None:
        L = int((len(params) - 1)/4)

    theta = params[0]
    mu_single = np.array(params[1:1 + L])
    sigma_single = np.array(params[1 + L:1 + 2 * L])
    mu_multiple = np.array(params[1 + 2 * L:1 + 3 * L])
    sigma_multiple = np.array(params[1 + 3 * L:1 + 4 * L])

    return (theta, mu_single, sigma_single, mu_multiple, sigma_multiple)


def _pack_params(theta, mu_single, sigma_single, mu_multiple, sigma_multiple):
    return np.hstack([theta, mu_single, sigma_single, mu_multiple, sigma_multiple])




def nlp(params, y, L):
    return -ln_prob(y, L, *params)



def get_initialization_point(y):
    N, D = y.shape

    init_dict = dict(
        theta=0.5,
        mu_single=np.median(y, axis=0),
        sigma_single=0.1 * np.median(y, axis=0),
        sigma_multiple=0.1 * np.ones(D),
    )

    # mu_multiple is *highly* constrained. Select the mid-point between what is
    # OK:
    
    mu_multiple_ranges = np.array([
        np.log(init_dict["mu_single"]) + init_dict["sigma_multiple"]**2,
        np.log(init_dict["mu_single"] + 3 * init_dict["sigma_single"]) + pow(init_dict["sigma_multiple"], 2)
    ])

    init_dict["mu_multiple"] = np.mean(mu_multiple_ranges, axis=0)
    #init_dict["mu_multiple_uv"] = 0.5 * np.ones(D)

    x0 = _pack_params(**init_dict)
    
    op_kwds = dict(x0=x0, args=(y, D))

    p_opt = op.minimize(nlp, **op_kwds)

    # make no checks
    return p_opt.x
