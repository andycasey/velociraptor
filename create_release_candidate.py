
"""
Fit a non-parametric model to the jitter in radial velocity error and the
astrometric unit weight error, and then use a gaussian process to enforce
smoothness and make predictions for 10^7 sources.
"""

import george
import logging as logger
import multiprocessing as mp
import numpy as np
import os
import pickle
import tqdm
import yaml
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict
from george import kernels
from scipy import optimize as op
from scipy.special import logsumexp
from time import (sleep, time) # yea it is

import npm_utils as npm
import stan_utils as stan


with open("model.yaml", "r") as fp:
    config = yaml.load(fp)

results_path = config["results_path"]
if os.path.exists(results_path) and not config.get("overwrite_results", False):
    raise IOError("{} exists and we will not overwrite".format(results_path))

np.random.seed(config.get("random_seed", None))

# Load the data.
data = fits.open(config["data_path"])[config.get("data_hdu", 1)].data

catalog_results = OrderedDict()

# Select the sources that will be used by the non-parametric model and the
# gaussian process. These sources should have:
# - finite values in the predictor label names of all non-parametric models
# - finite values in the kdtree label names of all non-parametric models

relevant_label_names = list(set(sum([
    (m["predictor_label_names"], m["kdtree_label_names"]) \
    for descr, m in config["non_parametric_models"].items()])))

is_suitable_source = np.all(np.isfinite(np.array(
    [data[ln] for ln in relevant_label_names])), axis=0)

logger.info("There are {} suitable sources".format(sum(is_suitable_source)))

# We will use all indices relative to data_subset, because if a source does not
# have finite properties of some label then we cannot use it for GP predictions
# anyways.
data_subset = data[is_suitable_source]

# Select the sources that will be used for fitting.
S = len(data_subset)
N = min(config["number_of_sources_to_fit"], S)

finite_indices = np.arange(S)
data_indices = np.random.choice(finite_indices, size=N, replace=False)


def construct_kernel(D):
    return kernels.ExpKernel(np.ones(D), ndim=D) \
         + kernels.Matern32Kernel(np.ones(D), ndim=D)


def get_reference_indices(indices):
    indices = np.atleast_1d(indices)
    ai = np.where(np.in1d(data_indices, indices))[0]
    foo =  ai[np.argsort(indices)]

    assert np.all(data_indices[foo] == indices)
    return foo

gp_results = dict()
npm_results = dict()

# Run a non-parametric model and the gaussian process for that model.
for description, npm_config in config["non_parametric_models"].items():

    logger.info("Setting up {} non-parametric model".format(description))

    model = stan.load_stan_model(npm_config["model_path"], verbose=False)

    # Set up the KD-tree for all finite sources.
    X = np.vstack([data_subset[ln] for ln in npm_config["kdtree_label_names"]]).T
    kdt, kdt_scales, kdt_offsets = \
        npm.build_kdtree(X, relative_scales=npm_config["kdtree_relative_scales"])

    kdt_kwds = dict(
        offsets=kdt_offsets, 
        scales=kdt_scales,
        full_output=True,
        minimum_radius=npm_config.get("kdtree_minimum_radius", None),
        maximum_radius=npm_config.get("kdtree_maximum_radius", None),
        minimum_points=npm_config.get("kdtree_minimum_points", None),
        maximum_points=npm_config.get("kdtree_maximum_points", None),
        minimum_density=npm_config.get("kdtree_minimum_density", None))

    # Set up the optimisation keywords.
    opt_kwds = {**{}, **npm_config.get("optimisation_kwds", {})}
    for opt_key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
        if opt_key in opt_kwds:
            opt_kwds[opt_key] = float(opt_kwds[opt_key])
    
    # Initialisation point.
    init_dict = npm._check_params_dict(npm_config.get("initialisation", None))

    bounds_dict = npm_config.get("parameter_bounds", None)

    # Arrays for results.
    L = 4 * len(npm_config["predictor_label_names"]) + 1
    
    done = np.zeros(N, dtype=bool)
    queued = np.zeros(N, dtype=bool)
    results = np.nan * np.ones((N, L), dtype=float)


    def optimise_mixture_model(index, init=None, boundary_tolerance=1e-2):

        d, idx, meta = npm.query_around_point(kdt, X[index], **kdt_kwds)

        y = np.array([data_subset[ln][idx] \
                      for ln in npm_config["predictor_label_names"]]).T

        if init is None:
            init = npm.get_initialization_point(y)

        # TODO: Add debug information about the ball properties?
        meta_dict = dict()
        data_dict = dict(y=y, N=y.shape[0], D=y.shape[1])

        # TODO: Check that these are implemented.
        for key, value in bounds_dict.items():
            data_dict["{}_bounds".format(key)] = value

        # If the optimization fails from the initial point then we should re-run
        # it from a random point.
        results = []
        for j, trial_init in enumerate((init, "random")):

            if isinstance(trial_init, dict) and bounds_dict is not None:
                trial_init = npm._check_params_dict(trial_init, 
                                                    bounds_dict=bounds_dict,
                                                    fail_on_bounds=False,
                                                    tolerance=0.01)

            kwds = dict(init=trial_init, data=data_dict)
            kwds.update(opt_kwds)

            outputs = []
            with stan.suppress_output() as sm:
                try:
                    p_opt = model.optimizing(**kwds)

                except:
                    p_opt = None
                    # TODO: Warn about relxing optimization tolerances?
                    # Save the outputs for debugging purposes before we close sm
                    outputs.extend([sm.stdout, sm.stderr])

            if p_opt is None:
                stdout, stderr = outputs

                logger.warning("Exception when optimizing {} model on index {}"\
                               "from point {}".format(
                                    description, index, trial_init))
                logger.warning("Stan stdout:\n{}".format(stdout))
                logger.warning("Stan stderr:\n{}".format(stderr))

            else:
                # Check that it's not bunched up against an edge of the bounds.
                for k, v in p_opt.items():
                    if k not  in bounds_dict \
                    or (k == "theta" and (v >= 1 - boundary_tolerance)):
                        continue

                    lower, upper = bounds_dict[k]
                    if np.abs(v - lower) <= boundary_tolerance \
                    or np.abs(v - upper) <= boundary_tolerance:
                        logger.warning("Optimised {} {} at edge of grid "\
                                       "({} < {} < {}) - ignoring".format(
                                        description, k, lower, v, upper))

                        break
                else:
                    break

        else:
            logger.warning("Optimization on index of {} did not converge from "\
                           "any point trialled. Consider relaxing optimisation"\
                           " tolerances!".format(index))
            p_opt = None


        p_opt = npm._check_params_dict(p_opt)

        return (index, p_opt, idx, meta)



    def serial_process(*indices, **kwargs):

        logger.info("Running process in serial")

        C = npm_config.get("assign_optimised_result_to_nearest", 0)

        with tqdm.tqdm(indices, total=len(indices)) as pbar:

            for index in indices:

                _, result, kdt_indices, meta = optimise_mixture_model(index,
                                                                      init_dict)

                pbar.update()

                reference_index = get_reference_indices(index)

                done[reference_index] = True
                
                if result is not None:

                    results[reference_index] = npm._pack_params(**result)

                    if C > 0:
                        nearby_indices = get_reference_indices(kdt_indices[:C + 1])

                        done[nearby_indices] = True
                        results[nearby_indices] = npm._pack_params(**result)

                        updated = nearby_indices.size - sum(done[nearby_indices])
                        pbar.update(updated)

        return None


    def parallel_process(*_, in_queue=None, candidate_queue=None, out_queue=None,
        seed=None):

        C = npm_config.get("assign_optimised_result_to_nearest", 0)
        B = npm_config.get("use_optimised_result_as_initialisation_for_nearest", 0)

        while True:

            try:
                index, init = in_queue.get_nowait()

            except mp.queues.Empty:
                logger.info("Swarm is bored.")
                break

            except:
                logger.exception("Exception:")
                logger.exception("Killing thread...")
                break

            else:
                try:
                    _, result, kdt_idx, meta = optimise_mixture_model(index, init)

                except:
                    logger.exception("Exception when optimizing index {} from {}"\
                                     .format(index, init))
                    continue

                out_queue.put((index, result, meta))

                if result is not None:
                    if B > 0:
                        # Assign closest points to have the same result.
                        out_queue.put((kdt_idx[:B + 1], result))

                    if C > 0:
                        # Use result as initial point for nearby points.
                        candidate_queue.put((kdt_idx[B + 1:B + 1 + C], result))


        return None

    P = npm_config.get("multiprocessing_threads", -1)
    P = P if P >= 0 else mp.cpu_count()

    if P > 1:

        # Parallel.
        min_in_queue = 10 * P

        with mp.Pool(processes=P) as pool:

            manager = mp.Manager()
            in_queue = manager.Queue()
            candidate_queue = manager.Queue()
            out_queue = manager.Queue()

            swarm_initialised = False
            swarm_kwds = dict(in_queue=in_queue,
                              out_queue=out_queue,
                              candidate_queue=candidate_queue)

            with tqdm.tqdm(total=N) as pbar:

                while True:            

                    # Check for candidates.
                    try:
                        r = candidate_queue.get_nowait()

                    except mp.queues.Empty:
                        None

                    else:
                        candidate_idx, init = r
                        candidate_ref_idx = get_reference_indices(candidate_idx)

                        for index in candidate_ref_idx:
                            if not done[index] and not queued[index]:
                                in_queue.put((index, init))
                                queued[index] = True

                    # Always make sure there is enough in the in_queue.
                    Q = in_queue.qsize()
                    if Q < min_in_queue:
                        not_done = data_indices[np.where(~done)[0]]
                        V = min(min_in_queue, not_done.size)
                        if V > 0:
                            for idx in np.random.choice(not_done, V, False):
                                in_queue.put((idx, init_dict))

                    # Initialise the swarm, if it isn't already.
                    if not swarm_initialised:
                        swarm_initialised = True
                        for _ in range(P):
                            pool.apply_async(parallel_process, [], kwds=swarm_kwds)


                    # Check for outputs.
                    try:
                        r = out_queue.get(timeout=npm_config.get("timeout", 5))

                    except mp.queues.Empty:
                        if all(done):
                            break                       

                    else:
                        index, result, meta = r
                        reference_index = get_reference_indices(index)

                        updated = reference_index.size - sum(done[reference_index])
                        done[reference_index] = True

                        if result is not None:
                            results[reference_index] = npm._pack_params(**result)

                        pbar.update(updated)

    else:
        # Serial.
        serial_process(*data_indices)

    prefix = npm_config["descriptive_prefix"]
    gp_labels = (
    	"npm_{prefix}_theta",
        "npm_{prefix}_mu_single",
        "npm_{prefix}_sigma_single",
        "npm_{prefix}_mu_multiple",
        "npm_{prefix}_sigma_multiple",
    )

    for i, gp_label_format in enumerate(gp_labels):
    	_ = np.nan * np.ones(len(data))
    	_[data_indices] = results[:, i]
    	catalog_results[gp_label_format.format(prefix=prefix)] = _

    # Run gaussian process on results.
    use_in_gp = np.all(np.isfinite(results), axis=1)

    x = X[data_indices][use_in_gp]
    _, D = x.shape
    
    # TODO: revisit this.
    x_for_gp = lambda x: x

    chunk_size = npm_config.get("chunk_size", 10000)
    gp_results[description] = []

    npm_labels = (
        "{prefix}_mu_single",
        "{prefix}_sigma_single",
        "{prefix}_mu_multiple",
        "{prefix}_sigma_multiple",
    )

    x_pred = np.vstack([data[ln] for ln in npm_config["kdtree_label_names"]]).T
    W, _ = x_pred.shape

    pred_indices = np.arange(W)[np.all(np.isfinite(x_pred), axis=1)]

    for npm_index, npm_label_format in enumerate(npm_labels, start=1):

        npm_label = npm_label_format.format(prefix=prefix)

        logger.info("Fitting Gaussian process to {} model index {}: {}".format(
                    description, npm_index, npm_label))

        y = results[use_in_gp, npm_index]

        kernel = construct_kernel(D)

        gp = george.GP(kernel,
                       mean=np.mean(y), white_noise=np.log(np.std(y)),
                       fit_mean=True, fit_white_noise=True)

        def nll(p):
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(y, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        def grad_nll(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y, quiet=True)

        gp.compute(x_for_gp(x))

        logger.info("Initial \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
        logger.info("Initial \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

        p0 = gp.get_parameter_vector()

        t_init = time()
        gp_result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
        t_opt = time() - t_init

        logger.info("Result: {}".format(gp_result))
        logger.info("Final \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
        logger.info("Took {:.0f} seconds to optimize".format(t_opt))

        gp_results[description].append(gp)


        # Make predictions for all sources.
        logger.info("Making {} predictions for all sources".format(npm_label))

        catalog_results.setdefault(npm_label, np.nan * np.ones((W, 2)))

        H = int(np.ceil(pred_indices.size / float(chunk_size)))
        with tqdm.tqdm(total=pred_indices.size) as pbar:
            for h in range(1 + H):
                chunk_indices = pred_indices[h * chunk_size:(h + 1) * chunk_size]

                pred, pred_var = gp.predict(y, x_for_gp(x_pred[chunk_indices]),
                                            return_var=True)

                catalog_results[npm_label][chunk_indices, 0] = pred
                catalog_results[npm_label][chunk_indices, 1] = pred_var

                pbar.update(chunk_indices.size)

    # Calculate log-likelihoods given GP parameters.
    y = np.vstack([data[ln] for ln in npm_config["predictor_label_names"]]).T[0]

    mu = catalog_results[f"{prefix}_mu_single"].T[0]
    sigma = catalog_results[f"{prefix}_sigma_single"].T[0]

    catalog_results[f"{prefix}_ln_likelihood_single"] = \
        -0.5 * np.log(2*np.pi) - np.log(sigma) \
        -0.5 * ((y - mu)/sigma)**2

    mu = catalog_results[f"{prefix}_mu_multiple"].T[0]
    sigma = catalog_results[f"{prefix}_sigma_multiple"].T[0]

    catalog_results[f"{prefix}_ln_likelihood_multiple"] = \
        -0.5 * np.log(2*np.pi) - np.log(y * sigma) \
        -0.5 * ((np.log(y) - mu)/sigma)**2

    ln_likelihoods = np.vstack([
        catalog_results[f"{prefix}_ln_likelihood_single"],
        catalog_results[f"{prefix}_ln_likelihood_multiple"]
    ]).T


    ln_likelihood = logsumexp(ln_likelihoods, axis=1)

    with np.errstate(under="ignore"):
        log_tau_single = ln_likelihoods.T[0] - ln_likelihood

    catalog_results[f"{prefix}_tau_single"] = np.exp(log_tau_single)

# TODO: Calculate joint likelihoods and tau.
metadata_path = config["metadata_path"]
logger.info("Saving metadata to {}".format(metadata_path))

with open(metadata_path, "wb") as fp:
    pickle.dump((config, gp_results), fp)

# Save the catalog.
results_path = config["results_path"]
logger.info("Saving catalog to {}".format(results_path))

del data
data = Table.read(results_path)
for k, v in catalog_results.items():
    data[k] = v

data.write(results_path, overwrite=True)


