

"""
Non-parametric model of binarity and single stars across the H-R diagram.
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from scipy import (spatial, optimize as op)
from sklearn import neighbors as neighbours

import velociraptor
import stan_utils as stan

np.random.seed(42)


# 
data_path = "data/rv-all-subset-1e4.fits"
data = fits.open(data_path)[1].data


# [1] Construct KD-Tree
# [2] Select a random X number of stars
# [3] Construct some way to get nearest neighbour results that are not Null???
#       --> select the next nearest star? or one star in the volume that does not have results?
# [3] Run the optimization etc.


# what labels are we going to build the KD-tree in?
kdt_label_names = ("bp_rp", "absolute_rp_mag", "phot_rp_mean_mag")

predictor_label_names = (
    "rv_single_epoch_variance",

)
#
#    "rv_abs_diff_template_teff",
#    "astrometric_unit_weight_error",
#    "phot_bp_rp_excess_factor",
#)


X_kdt = np.vstack([data[ln] for ln in kdt_label_names]).T
# TODO: Right now I am *requiring* that all predictor labels and KD-Tree
#       labels are finite, but we may want to change this in the future.
all_label_names = tuple(list(kdt_label_names) + list(predictor_label_names))
subset = np.all(np.isfinite(np.vstack([data[ln] for ln in all_label_names])), axis=0)
X_kdt = X_kdt[subset]
X_scale = np.ptp(X_kdt, axis=0)
X_mean = np.mean(X_kdt, axis=0)

_scale = lambda a: (a - X_mean)/X_scale
_descale = lambda a: a * X_scale + X_mean

# Normalise the array for the KD-tree
X_norm = _scale(X_kdt)

# Construct the KD-Tree
kdt_kwds = dict(leaf_size=40, metric="minkowski")
kdt = neighbours.KDTree(X_norm, **kdt_kwds)

data = data[subset]


def get_ball_around_point(index, K=1000, full_output=False):
    dist, k_indices = kdt.query(X_norm[[index]], K)
    dist, k_indices = (dist[0], k_indices[0])
    return (k_indices, dist) if full_output else k_indices


model = stan.load_stan_model("npm.stan")

# Calculate the total number of parameters
M = len(data)
K = 1 + 2 * len(predictor_label_names)


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
    min_mu_multiple = np.log(s_mu) + b_sigma**2 + s_sigma

    if not (1 >= theta >= 0) \
    or np.any(s_mu <= 0) \
    or np.any(s_sigma <= 0) \
    or np.any(b_sigma <= 0) \
    or np.any(b_mu <= min_mu_multiple):
        return -np.inf
    return 0


def ln_prob(y, L, *params):
    theta, s_mu, s_sigma, b_mu, b_sigma = _unpack_params(params, L=L)
    lp = ln_prior(theta, s_mu, s_sigma, b_mu, b_sigma)
    if np.isfinite(lp):
        return lp + ln_likelihood(y, theta, s_mu, s_sigma, b_mu, b_sigma)
    return lp


def _unpack_params(params, L):
    # unpack the multdimensional values.
    theta = params[0]
    mu_single = np.array(params[1:1 + L])
    sigma_single = np.array(params[1 + L:1 + 2 * L])
    mu_multiple = np.array(params[1 + 2 * L:1 + 3 * L])
    sigma_multiple = np.array(params[1 + 3 * L:1 + 4 * L])

    return (theta, mu_single, sigma_single, mu_multiple, sigma_multiple)


def _pack_params(theta, mu_single, sigma_single, mu_multiple, sigma_multiple):
    return np.hstack([theta, mu_single, sigma_multiple, mu_multiple, sigma_multiple])




def nlp(params, y, L):
    return -ln_prob(y, L, *params)



def get_initialization_point(y):
    N, D = y.shape

    init_dict = dict(
        theta=0.5,
        mu_single=np.median(y, axis=0),
        sigma_single=0.1 * np.median(y, axis=0),
        sigma_multiple=1e-2 * np.ones(D),
    )
    init_dict.update(
        mu_multiple=1e-3 + np.log(init_dict["mu_single"]) \
                    + init_dict["sigma_multiple"]**2 \
                    + init_dict["sigma_single"])

    x0 = _pack_params(**init_dict)
    L = len(predictor_label_names)

    op_kwds = dict(x0=x0, args=(y, L))

    p_opt = op.minimize(nlp, **op_kwds)

    # make no checks
    return p_opt.x



init_theta = 0.5
init_sigma_multiple = 0.1
mu_multiple_factor = 1. + 1e-2

opt_params = np.empty((M, K))

parameter_names = ["theta", "mu_single", "sigma_single", "mu_multiple",
    "sigma_multiple"]

for i in range(M):

    k_indices, dist = get_ball_around_point(i, full_output=True)

    y = np.array([data[ln][k_indices] for ln in predictor_label_names]).T

    L = len(predictor_label_names)

    init_values = get_initialization_point(y)
    init_dict = dict(zip(parameter_names, _unpack_params(init_values, L)))

    fig, axes = plt.subplots(1, L)
    axes = np.atleast_1d(axes).flatten()

    for j, ax in enumerate(axes):
        ax.hist(y.T[j])

        #ax.set_title(predictor_label_names[j])

    N, D = y.shape
    opt_kwds = dict(
        data=dict(y=y, N=N, D=D, max_y=np.max(y, axis=0)),
        init=init_dict,
        iter=100000, 
        tol_obj=7./3 - 4./3 - 1, # machine precision
        tol_grad=7./3 - 4./3 - 1, # machine precision
        tol_rel_grad=1e3,
        tol_rel_obj=1e4)



    p_opt = model.optimizing(**opt_kwds)
    for k in p_opt.keys():
        if k == "theta":  continue
        p_opt[k] = np.atleast_1d(p_opt[k])

    samples  = model.sampling(**stan.sampling_kwds(
        data=opt_kwds["data"], init=p_opt, iter=10000, chains=2))

    chains_dict = samples.extract()
    if L == 1:
        chains  = np.vstack([
            chains_dict["theta"],
            chains_dict["mu_single"],
            chains_dict["sigma_single"],
            chains_dict["mu_multiple"],
            chains_dict["sigma_multiple"]
        ]).T

    else:
        chains  = np.hstack([
            np.atleast_2d(chains_dict["theta"]).T,
            chains_dict["mu_single"],
            chains_dict["sigma_single"],
            chains_dict["mu_multiple"],
            chains_dict["sigma_multiple"]
        ])


    # Make plots of the pdf of each distribution.

    def norm_pdf(x, norm_mu, norm_sigma, theta):
        return theta * (2 * np.pi * norm_sigma**2)**(-0.5) * np.exp(-(x - norm_mu)**2/(2*norm_sigma**2))
        
    def lognorm_pdf(x, lognorm_mu, lognorm_sigma, theta):
        return (1.0 - theta)/(x * lognorm_sigma * np.sqrt(2*np.pi)) \
               * np.exp(-0.5 * ((np.log(x) - lognorm_mu)/lognorm_sigma)**2)

    xi = np.linspace(np.min(y.T[0]), np.max(y.T[0]), 1000)

    indices = np.random.choice(len(chains), 100, replace=False)
        
    fig, ax = plt.subplots()
    for index in indices:
        
        idx_theta = parameter_names.index("theta")
        idx_norm_mu = parameter_names.index("mu_single")
        idx_norm_sigma = parameter_names.index("sigma_single")
        idx_lognorm_mu = parameter_names.index("mu_multiple")
        idx_lognorm_sigma = parameter_names.index("sigma_multiple")
        
        theta = chains[index, idx_theta]
        norm_mu = chains[index, idx_norm_mu]
        norm_sigma = chains[index, idx_norm_sigma]
        lognorm_mu = chains[index, idx_lognorm_mu]
        lognorm_sigma = chains[index, idx_lognorm_sigma]
        
        ax.plot(xi, norm_pdf(xi, norm_mu, norm_sigma, theta), c='r', alpha=0.1)
        ax.plot(xi, lognorm_pdf(xi, lognorm_mu, lognorm_sigma, theta), c='b', alpha=0.1)
        
    _ = ax.hist(y, bins=500, facecolor="#000000", zorder=-1, normed=True)
    print(y.size)

    # = model.sampling(**stan.sampling_kwds(data=data_dict, init=init_dict, iter=2000))
    raise a


