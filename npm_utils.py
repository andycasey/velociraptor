
import numpy as np
import os
from scipy import (optimize as op, stats)
from sklearn import neighbors as neighbours
from scipy.special import logsumexp
from time import time
import warnings

import stan_utils as stan

CONFIG_PATH = "npm-config.yaml"

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




def query_around_point(kdtree, point, offsets=0, scales=1, minimum_radius=None, 
    maximum_radius=None, minimum_points=1, maximum_points=None, 
    minimum_density=None, dualtree=False,
    full_output=False, **kwargs):
    """
    Query around a point in the KD-Tree until certain conditions are met (e.g.,
    the number of points in the ball, and the minimum radius that the ball
    expands out to).
    
    :param kdtree:
        The pre-computed KD-Tree.

    :param point:
        The (unscaled) point to query around.

    :param offsets: [optional]
        The offsets to apply to the query point.

    :param scales: [optional]
        The scaling to apply to the query point, after subtracting the offsets.

    :param minimum_radius: [optional]
        The minimum radius (or radii) that the ball must extend to.

    :param minimum_points: [optional]
        The minimum number of points to return in the ball.

    :param maximum_points: [optional]
        The maximum number of points to return in the ball. If the number of
        points returned exceeds this value, then a random subset of the points
        will be returned.

    :param minimum_density: [optional]
        The minimum average density of points per dimension for the ball. This
        can be useful to ensure that points that are in the edge of the k-d tree
        parameter space will be compared against points that are representative
        of the underlying space, and not just compared against nearest outliers.

    :param dualtree: [optional]
        Use the dual tree formalism for the query: a tree is built for the query
        points, and the pair of trees is  used to efficiently search this space.
        This can lead to better performance as the number of points grows large.

    :param full_output: [optional]
        If `True`, return a two length tuple of the distances to each point and
        the indicies, otherwise just return the indices.
    """

    #print("querying k-d tree")

    offsets = np.atleast_1d(offsets)
    scales = np.atleast_1d(scales)

    point_orig = np.atleast_1d(point).reshape(1, -1)
    point = (point_orig - offsets)/scales

    # Simple case.
    if minimum_radius is None and minimum_density is None:
        # We can just query the nearest number of points.
        d, indices = kdtree.query(point, k=minimum_points, 
            sort_results=True, return_distance=True, dualtree=dualtree)

    else:
        # We need to find the minimum radius that meets our constraints.
        if minimum_radius is None: 
            minimum_radius = 0

        if minimum_density is None:
            minimum_density = 0

        minimum_radius = np.atleast_1d(minimum_radius)
        minimum_density = np.atleast_1d(minimum_density)
        
        # Need to scale the minimum radius from the label space to the normalised
        # k-d tree space.
        minimum_radius_norm = np.max(minimum_radius / np.atleast_1d(scales))

        K = kdtree.two_point_correlation(point, minimum_radius_norm)[0]

        # "density" = N/(2*R)
        # if N > 2 * R * density then our density constraint is met
        K_min = np.max(np.hstack([
            minimum_points, 
            2 * minimum_density * minimum_radius
        ]))

        # Check that the minimum radius norm will also meet our minimum number
        # of points constraint. Otherwise, we need to use two point
        # auto-correlation functions to see how far to go out to.
        if K >= K_min:
            # All constraints met.
            radius_norm = minimum_radius_norm

        else:
            #print("Using k-dtree to step out in radius because we have {} points within {} but need {}".format(
            #    K, minimum_radius_norm, K_min))

            # We need to use the k-d tree to step out until our constraints are
            # met.
            maximum_radius_norm = 2 * np.max(np.ptp(kdtree.data, axis=0))

            # This is the initial coarse search.
            N, D = kdtree.data.shape
            left, right = (minimum_radius_norm, maximum_radius_norm)

            Q = kwargs.get("Q", 10) # MAGIC HACK
 
            # MAGIC HACK
            tolerance = maximum_points if maximum_points is not None \
                                       else 2 * minimum_points

            while True:
                # Shrink it.

                ri = np.logspace(np.log10(left), np.log10(right), Q)


                counts = kdtree.two_point_correlation(point, ri)
                #print("tolerance {}: {} {} {} {}".format(tolerance, left, right, ri, counts))



                minimum_counts = np.clip(2 * np.max(np.dot(ri.reshape(-1, 1), 
                    (minimum_density * scales).reshape(1, -1)), axis=1),
                    minimum_points, N)

                indices = np.arange(Q)[counts >= minimum_counts]

                left, right = (ri[indices[0] - 1], ri[indices[0] + 1])
                #print("new: {} {}".format(left, right))


                if np.diff(counts[indices]).max() < tolerance:
                    radius_norm = ri[indices[0]]
                    break

        # two_point_correlation(point, minimum_radius_norm)
        #   is eequivalent to
        # query_radius(point, minimum_radius_norm, count_only=True)
        # but in my tests two_point_correlation was a little faster.

        # kdtree.query_radius returns indices, d
        # kdtree.query returns d, indices
        # .... are you serious?

        indices, d = kdtree.query_radius(point, radius_norm, 
            return_distance=True, sort_results=True)


    d, indices = (d[0], indices[0])

    L = len(indices)
    if maximum_points is not None and L > maximum_points:
        if maximum_points < minimum_points:
            raise ValueError("minimum_points must be smaller than maximum_points")

        if maximum_radius is not None:
            L = np.where(np.all(np.abs(point[0] - np.asarray(kdtree.data)[indices]) <= maximum_radius, axis=1))[0]
            maximum_points = min(maximum_points, L.size)

        # Sub-sample a random number.
        sub_idx = np.random.choice(L, maximum_points, replace=False)

        d, indices = (d[sub_idx], indices[sub_idx])


    elif maximum_radius is not None:

        L = np.where(np.all(np.abs(point[0] - np.asarray(kdtree.data)[indices]) <= maximum_radius, axis=1))[0]
        maximum_points = min(maximum_points, L.size)

        # Sub-sample a random number.
        sub_idx = np.random.choice(L, maximum_points, replace=False)
        d, indices = (d[sub_idx], indices[sub_idx])


    # Meta should include the PTP values of points in the ball.
    meta = dict()

    #assert minimum_points is None or indices.size >= minimum_points
    return (d, indices, meta) if full_output else indices






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
    
    s_lpdf = -hl2p + 0.5 * np.log(s_ivar) \
           - 0.5 * (y - s_mu)**2 * s_ivar
    
    b_lpdf = -np.log(y*b_sigma) - hl2p \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar

    foo = np.vstack([s_lpdf, b_lpdf]).T + np.log([theta, 1-theta])
    ll = np.sum(logsumexp(foo, axis=1))

    #ll = np.sum(s_lpdf) + np.sum(b_lpdf)

    ##print(lpdf)
    
    #assert np.isfinite(ll)
    return ll


def ln_prior(theta, s_mu, s_sigma, b_mu, b_sigma):

    # Ensure that the *mode* of the log-normal distribution is larger than the
    # mean of the normal distribution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_mu_multiple = np.log(s_mu + s_sigma) + b_sigma**2

    if not (1 >= theta >= 0) \
    or not (15 >= s_mu >= 0.5) \
    or not (10 >= s_sigma >= 0.05) \
    or not (1.6 >= b_sigma >= 0.20) \
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


def _pack_params(theta, mu_single, sigma_single, mu_multiple, sigma_multiple, mu_multiple_uv=None, **kwargs):
    if mu_multiple_uv is None:
        return np.hstack([theta, mu_single, sigma_single, mu_multiple, sigma_multiple])
    else:
        return np.hstack([theta, mu_single, sigma_single, mu_multiple, sigma_multiple, mu_multiple_uv])


def _check_params_dict(d, bounds_dict=None, fail_on_bounds=True, tolerance=0.01):
    if d is None: return d
    
    dc = {**d}
    for k in ("mu_single", "sigma_single", "mu_multiple", "sigma_multiple"):
        dc[k] = np.atleast_1d(dc[k]).flatten()[0]
        if bounds_dict is not None and k in bounds_dict:
            lower, upper = bounds_dict[k]
            if (not np.all(upper >= dc[k]) or not np.all(dc[k] >= lower)):
                if fail_on_bounds:
                    raise ValueError("bounds not met: {} = {} not within ({} {})"\
                                     .format(k, dc[k], lower, upper))
                else:
                    print("Clipping initial {} to be within bounds ({}, {}): {}"\
                        .format(k, lower, upper, dc[k]))
                    dc[k] = np.clip(dc[k], lower + tolerance, upper - tolerance)


    return dc


def nlp(params, y, L):
    return -ln_prob(y, L, *params)


def _get_1d_initialisation_point(y, scalar=5, bounds=None):

    N= y.size

    init = dict(
        theta=0.75,
        mu_single=np.min([np.median(y, axis=0), 10]),
        sigma_single=0.2,
        sigma_multiple=0.5)

    if bounds is not None:
        for k, (lower, upper) in bounds.items():
            if not (upper >= init[k] >= lower):
                init[k] = np.mean([upper, lower])

    lower_mu_multiple = np.log(init["mu_single"] + scalar * init["sigma_single"]) \
                      + init["sigma_multiple"]**2

    init["mu_multiple"] = 1.1 * lower_mu_multiple


    op_kwds = dict(x0=_pack_params(**init), args=(y, 1))

    nlp = lambda params, y, L: -ln_prob(y, L, *params)
    p_opt = op.minimize(nlp, **op_kwds)

    keys = ("theta", "mu_single", "sigma_single", "mu_multiple", "sigma_multiple")

    init_dict = _check_params_dict(init)
    op_dict = _check_params_dict(dict(zip(keys, _unpack_params(p_opt.x))))

    # Only return valid init values.
    valid_inits = []
    for init in (init_dict, op_dict):
        if np.isfinite(nlp(_pack_params(**init), y, 1)):
            valid_inits.append(init)

    valid_inits.append("random")

    return valid_inits


def get_initialization_point(y):
    N, D = y.shape

    ok = y <= np.mean(y)

    init_dict = dict(
        theta=0.5,
        mu_single=np.median(y[ok], axis=0),
        sigma_single=0.1 * np.median(y[ok], axis=0),
        sigma_multiple=0.1 * np.ones(D),
    )



    # mu_multiple is *highly* constrained. Select the mid-point between what is
    # OK:
    mu_multiple_ranges = np.array([
        np.log(init_dict["mu_single"] + 1 * init_dict["sigma_single"]) + init_dict["sigma_multiple"]**2,
        np.log(init_dict["mu_single"] + 5 * init_dict["sigma_single"]) + pow(init_dict["sigma_multiple"], 2)
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

    init_dict = _check_params_dict(init_dict)

    return init_dict
