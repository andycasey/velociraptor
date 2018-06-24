

"""
Non-parametric model of binarity and single stars across the H-R diagram.
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from astropy.table import Table
from astropy.io import fits
from scipy import (spatial, optimize as op)
from sklearn import neighbors as neighbours
from time import time

import velociraptor
import stan_utils as stan
import npm_utils

from corner import corner

np.random.seed(123)

DEBUG_PLOTS = False

data_path = "data/rv-all-subset-1e4.fits"
#data = fits.open(data_path)[1].data
data = Table.read(data_path)
data["rv_single_epoch_scatter"] = data["rv_single_epoch_variance"]**0.5



# [1] Construct KD-Tree
# [2] Select a random X number of stars
# [3] Construct some way to get nearest neighbour results that are not Null???
#       --> select the next nearest star? or one star in the volume that does not have results?
# [3] Run the optimization etc.


# what labels are we going to build the KD-tree in?
kdt_label_names = ("bp_rp", "absolute_rp_mag", "phot_rp_mean_mag")


predictor_label_names = (
    "rv_single_epoch_scatter",
    "astrometric_unit_weight_error",
    "phot_bp_rp_excess_factor",
    "rv_abs_diff_template_teff",
)

parameter_names = ["theta", "mu_single", "sigma_single", "mu_multiple",
    "sigma_multiple"]


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
kdt = npm_utils.build_kdt(X_norm)

data = data[subset]

model = stan.load_stan_model("npm.stan")


# Calculate the total number of parameters
M = len(data)
L = len(predictor_label_names)
K = 1 + 4 * L

opt_params = np.empty((M, K))

#subset_points = np.random.choice(M, size=1000, replace=False)
#for i in range(M):

#for j, i in enumerate(subset_points):
for j, i in enumerate(range(M)):

    print("At point {}/{}: {}".format(j, M, i))

    k_indices, dist = npm_utils.get_ball_around_point(kdt, X_norm[[i]],
                                                      full_output=True)

    y = np.array([data[ln][k_indices] for ln in predictor_label_names]).T
    N, D = y.shape
    
    init_values = npm_utils.get_initialization_point(y)
    init_dict = dict(zip(parameter_names, npm_utils._unpack_params(init_values, D)))  
    init_dict["mu_multiple_uv"] = 0.5 * np.ones(D)

    data_dict = dict(y=y, N=N, D=D)

    opt_kwds = dict(
        data=data_dict,
        init=init_dict,
        verbose=False,
        tol_obj=7./3 - 4./3 - 1, # machine precision
        tol_grad=7./3 - 4./3 - 1, # machine precision
        tol_rel_grad=1e3,
        tol_rel_obj=1e4,
        iter=10000)


    t_init = time()
    p_opt = model.optimizing(**opt_kwds)
    t_opt = time() - t_init

    print("Optimization took {:.2f} seconds".format(t_opt))
    
    del p_opt["mu_multiple_uv"]

    for k in p_opt.keys():
        if k == "theta":  continue
        p_opt[k] = np.atleast_1d(p_opt[k])

    opt_params[i] = npm_utils._pack_params(**p_opt)

    print("Single star fraction at this point: {:.2f}".format(opt_params[i, 0]))

    if DEBUG_PLOTS:

        print("Running DEBUG plots")

        for l, predictor_label_name in enumerate(predictor_label_names):

            fig, ax = plt.subplots()
            xi = np.linspace(
                np.min(data_dict["y"].T[l].flatten()), 
                np.max(data_dict["y"].T[l].flatten()),
                1000)

            ax.hist(data_dict["y"].T[l].flatten(), bins=50, facecolor="#cccccc", zorder=-1)


            show = init_dict
            ax.plot(xi, N * npm_utils.norm_pdf(xi, show["mu_single"][l], show["sigma_single"][l], show["theta"]), c='r')
            ax.plot(xi, N * npm_utils.lognorm_pdf(xi, show["mu_multiple"][l], show["sigma_multiple"][l], show["theta"]), c='b')

            show = p_opt
            ax.plot(xi, N * npm_utils.norm_pdf(xi, show["mu_single"][l], show["sigma_single"][l], show["theta"]), c='m')
            ax.plot(xi, N * npm_utils.lognorm_pdf(xi, show["mu_multiple"][l], show["sigma_multiple"][l], show["theta"]), c='y')
            
                
        samples  = model.sampling(**stan.sampling_kwds(
            data=opt_kwds["data"], init=p_opt, iter=2000, chains=2))

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


        fig = corner(chains)

        # Make plots of the pdf of each distribution.
        for l, predictor_label_name in enumerate(predictor_label_names):

            xi = np.linspace(np.min(y.T[l]), np.max(y.T[l]), 1000)

            indices = np.random.choice(len(chains), 100, replace=False)

            fig, ax = plt.subplots()
            for index in indices:
                
                idx_theta = parameter_names.index("theta")
                idx_norm_mu = parameter_names.index("mu_single")
                idx_norm_sigma = parameter_names.index("sigma_single")
                idx_lognorm_mu = parameter_names.index("mu_multiple")
                idx_lognorm_sigma = parameter_names.index("sigma_multiple")
                

                theta = chains[index, 0]
                norm_mu = chains[index, 1 + l]
                norm_sigma = chains[index, 1 + L + l]
                lognorm_mu = chains[index, 1 + 2*L + l]
                lognorm_sigma = chains[index, 1 + 3*L + l]

                ax.plot(xi, npm_utils.norm_pdf(xi, norm_mu, norm_sigma, theta), c='r', alpha=0.1)
                ax.plot(xi, npm_utils.lognorm_pdf(xi, lognorm_mu, lognorm_sigma, theta), c='b', alpha=0.1)
                
            _ = ax.hist(y.T[l], bins=500, facecolor="#000000", zorder=-1, normed=True)
            print(y.size)

            ax.set_title(predictor_label_name.replace("_", " "))


        raise a




with open("rv-all-subset-1e4-optimized-unconstrained.pickle", "wb") as fp: 
    pickle.dump(opt_params, fp, -1)



def normal_lpdf(y, mu, sigma):
    ivar = sigma**(-2)
    return 0.5 * (np.log(ivar) - np.log(2 * np.pi) - (y - mu)**2 * ivar)

def lognormal_lpdf(y, mu, sigma):
    ivar = sigma**(-2)
    return - 0.5 * np.log(2 * np.pi) - np.log(y * sigma) \
           - 0.5 * (np.log(y) - mu)**2 * ivar



from scipy.special import logsumexp

# Calculate log-probabilities for all of the stars we considered.
def membership_probability(y, p_opt):

    y = np.atleast_1d(y)
    theta, s_mu, s_sigma, m_mu, m_sigma = npm_utils._unpack_params(p_opt)

    assert s_mu.size == y.size, "The size of y should match the size of mu"


    D = y.size
    ln_prob = np.zeros((D, 2))
    for d in range(D):
        ln_prob[d] = [
            normal_lpdf(y[d], s_mu[d], s_sigma[d]),
            lognormal_lpdf(y[d], m_mu[d], m_sigma[d])
        ]

    # TODO: I am not certain that I am summing these log probabilities correctly

    sum_ln_prob = np.sum(ln_prob, axis=0) # per mixture
    ln_likelihood = logsumexp(sum_ln_prob)

    with np.errstate(under="ignore"):
        ln_membership = sum_ln_prob - ln_likelihood

    return np.exp(ln_membership)


y = np.array([data[ln] for ln in predictor_label_names]).T
N, D = y.shape

p_single = np.array([membership_probability(y[i], opt_params[i])[0] for i in range(N)])


# Now calculate the amount of excess RV variance

def rv_excess_scatter(y, p_opt, label_index):

    y = np.atleast_1d(y)
    _, s_mu, s_sigma, __, ___ = npm_utils._unpack_params(p_opt)

    assert s_mu.size == y.size, "The size of y should match the size of mu"

    rv_single_epoch_excess = np.sqrt(y[label_index]**2 - s_mu[label_index]**2)
    rv_single_epoch_significance = rv_single_epoch_excess/s_sigma[label_index]

    return (rv_single_epoch_excess, rv_single_epoch_significance)


li = list(predictor_label_names).index("rv_single_epoch_scatter")
rv_excess = np.array([
    rv_excess_scatter(y[i], opt_params[i], li) for i in range(N)])

# Remember that the probabilities of binarity take into acocunt more than just
# the radial velocity excess!

# Make a corner plot of the various properties? Coloured by probabilities?



with open("rv-all-subset-1e4-results.pickle", "wb") as fp: 
    pickle.dump((subset, opt_params, predictor_label_names, p_single, rv_excess), fp, -1)

