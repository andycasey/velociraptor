
import numpy as np
import os
from scipy import (optimize as op, stats)
from sklearn import neighbors as neighbours
from scipy.special import logsumexp
from time import time

import stan_utils as stan

CONFIG_PATH = "npm-config.yaml"

def split_source_id(source_id, chunk=4):
    return "/".join(map("".join, zip(*[iter(str(source_id))] * chunk)))



def get_indices_path(source_id, config):
    path = os.path.join(config["results_path"], 
        "{0}/indices".format(split_source_id(source_id)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_output_path(source_id, config, check_path_exists=True):
    results_suffix = config.get("results_suffix", None)
    path = os.path.join(config["results_path"], "{0}/{1}{2}".format(
        split_source_id(source_id), 
        "+".join(config["predictor_label_names"]),
        ".{}".format(results_suffix) if results_suffix is not None else ""))
    if check_path_exists:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    return path




def build_kdtree(X, relative_scales=None,**kwargs):
    """
    Build a KD-tree from the finite values in the given array.
    """

    offset = np.mean(X, axis=0)

    if relative_scales is None:
        # Whiten the data.
        relative_scales = np.ptp(X, axis=0)
    
    X_norm = (X - offset)/relative_scales

    kdt_kwds = dict(leaf_size=40, metric="minkowski")
    kdt_kwds.update(kwargs)
    kdt = neighbours.KDTree(X_norm, **kdt_kwds)

    return (kdt, relative_scales, offset)


def query_around_point(kdtree, point, offset=0, scale=1, minimum_radius=None, 
    minimum_points=1, maximum_points=None, dualtree=False, full_output=False):
    """
    Query around a point in the KD-Tree until certain conditions are met (e.g.,
    the number of points in the ball, and the minimum radius that the ball
    expands out to).
    
    :param kdtree:
        The pre-computed KD-Tree.

    :param point:
        The (unscaled) point to query around.

    :param offset: [optional]
        The offset to apply to the query point.

    :param scale: [optional]
        The scaling to apply to the query point, after subtracting the offset.

    :param minimum_points: [optional]
        The minimum number of points to return in the ball.

    :param minimum_radius: [optional]
        The minimum radius (or radii) that the ball must extend to.

    :param dualtree: [optional]
        Use the dual tree formalism for the query: a tree is built for the query
        points, and the pair of trees is  used to efficiently search this space.
        This can lead to better performance as the number of points grows large.

    :param full_output: [optional]
        If `True`, return a two length tuple of the distances to each point and
        the indicies, otherwise just return the indices.
    """

    point_orig = np.atleast_1d(point).reshape(1, -1)
    point = (point_orig - offset)/scale
    N, D = kdtree.data.shape

    if minimum_radius is None or np.max(minimum_radius) <= 0:
        # We can just query the nearest number of points.
        d, indices = kdtree.query(point, k=minimum_points, 
            sort_results=True, return_distance=True, dualtree=dualtree)

    else:
        # What (scaled) radius do we need to go out to?
        minimum_radius = np.atleast_1d(minimum_radius)
        if minimum_radius.size == 1:
            minimum_radius = np.ones(D) * minimum_radius

        else:
            assert minimum_radius.size == D, \
                "minimum radius dimensions must match the data dimensions"

        # What normalised scale is this?
        min_radius = np.max(minimum_radius/scale)

        # Check that the minimum radius norm will also meet our minimum number
        # of points constraint. Otherwise, we need to use two point
        # auto-correlation functions to see how far to go out to.
        K = kdtree.two_point_correlation(point, min_radius)
        if minimum_points is not None and K < minimum_points:

            # If the KD-tree is normalised correctly then the PTP of the data
            # should be unity in all dimensions. Therefore the maximum radius is
            # twice this.

            # Nope, don't assume!
            max_radius = 2 * np.max(np.ptp(kdtree.data, axis=0))

            if minimum_points >= N:
                radius = max_radius

            else:
                # Use the two-point autocorrelation function to figure out the
                # appropriate radius.
                left, right = (min_radius, max_radius)

                max_scale = int(np.log10(N - K))
                Q = None # 10 * max_scale
                for scale in np.logspace(-max_scale, 1, Q):
    
                    P = int(scale * (N - K))
                    ri = np.linspace(left, scale * right, P)
                    counts = kdtree.two_point_correlation(point, ri)

                    try:
                        radius = ri[counts >= minimum_points][0]

                    except:
                        continue

                    else:
                        break
        else:
            radius = min_radius

        # two_point_correlation(point, minimum_radius_norm)
        #   is eequivalent to
        # query_radius(point, minimum_radius_norm, count_only=True)
        # but in my tests two_point_correlation was a little faster.

        # kdtree.query_radius returns indices, d
        # kdtree.query returns d, indices
        # .... are you serious?
        indices, d = kdtree.query_radius(point, radius, 
            return_distance=True, sort_results=True)

    d, indices = (d[0], indices[0])

    L = len(indices)
    if maximum_points is not None and L > maximum_points:
        if maximum_points < minimum_points:
            raise ValueError("minimum_points must be smaller than maximum_points")

        # Sub-sample a random number.
        sub_idx = np.random.choice(L, maximum_points, replace=False)
        d, indices = (d[sub_idx], indices[sub_idx])

    return (d, indices) if full_output else indices


def normal_lpdf(y, mu, sigma):
    ivar = sigma**(-2)
    return 0.5 * (np.log(ivar) - np.log(2 * np.pi) - (y - mu)**2 * ivar)

def lognormal_lpdf(y, mu, sigma):
    ivar = sigma**(-2)
    return - 0.5 * np.log(2 * np.pi) - np.log(y * sigma) \
           - 0.5 * (np.log(y) - mu)**2 * ivar



# Calculate log-probabilities for all of the stars we considered.
def membership_probability(y, p_opt):

    y = np.atleast_1d(y)
    theta, s_mu, s_sigma, m_mu, m_sigma = _unpack_params(_pack_params(**p_opt))

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



def label_excess(y, p_opt, label_index):

    y = np.atleast_1d(y)
    _, s_mu, s_sigma, __, ___ = _unpack_params(_pack_params(**p_opt))

    assert s_mu.size == y.size, "The size of y should match the size of mu"

    excess = np.sqrt(y[label_index]**2 - s_mu[label_index]**2)
    significance = excess/s_sigma[label_index]

    return (excess, significance)




def build_kdt(X_norm, **kwargs):

    kdt_kwds = dict(leaf_size=40, metric="minkowski")
    kdt_kwds.update(kwargs)
    kdt = neighbours.KDTree(X_norm, **kdt_kwds)

    return kdt


def get_ball_around_point(kdt, point, K=1000, scale=1, offset=0, full_output=False):
    dist, k_indices = kdt.query((point - offset)/scale, K)
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


def _pack_params(theta, mu_single, sigma_single, mu_multiple, sigma_multiple, **kwargs):
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

    init_dict = dict(zip(
        ("theta", "mu_single", "sigma_single", "mu_multiple", "sigma_multiple"),
        _unpack_params(p_opt.x)))
    init_dict["mu_multiple_uv"] = 0.5 * np.ones(D)

    return init_dict
