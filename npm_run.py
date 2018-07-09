
""" Set up and run the non-parametric model. """

import numpy as np
import os
import multiprocessing as mp
import pickle
import yaml
import tqdm
import logging
from time import sleep
from astropy.io import fits

import npm_utils as npm
import stan_utils as stan

with open(npm.CONFIG_PATH, "r") as fp:
    config = yaml.load(fp)

# Load in the data.
data = fits.open(config["data_path"])[1].data

all_label_names = list(config["kdtree_label_names"]) \
                + list(config["predictor_label_names"])

# Set up a KD-tree.
X = np.vstack([data[ln] for ln in config["kdtree_label_names"]]).T
finite = np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0)
finite_indices = np.where(finite)[0]

N, D = X.shape
F = finite_indices.size
L = 4 * len(config["predictor_label_names"]) + 1
C = config["share_optimised_result_with_nearest"]

kdt, scales, offsets = npm.build_kdtree(
    X[finite], relative_scales=config["kdtree_relative_scales"])

kdt_kwds = dict(offsets=offsets, scales=scales, full_output=False)
kdt_kwds.update(
    minimum_radius=config["kdtree_minimum_radius"],
    minimum_points=config["kdtree_minimum_points"],
    maximum_points=config["kdtree_maximum_points"],
    minimum_density=config.get("kdtree_minimum_density", None))

model = stan.load_stan_model(config["model_path"], verbose=False)

default_opt_kwds = dict(
    verbose=False,
    tol_obj=7./3 - 4./3 - 1, # machine precision
    tol_grad=7./3 - 4./3 - 1, # machine precision
    tol_rel_grad=1e3,
    tol_rel_obj=1e4,
    iter=10000)
default_opt_kwds.update(config.get("optimisation_kwds", {}))

# Make sure that some entries have the right units.
for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
    if key in default_opt_kwds:
        default_opt_kwds[key] = float(default_opt_kwds[key])

logging.info("k-d tree keywords: {}".format(kdt_kwds))
logging.info("optimization keywords: {}".format(default_opt_kwds))


done = np.zeros(N, dtype=bool)
queued = np.zeros(N, dtype=bool)
results = np.nan * np.ones((N, L), dtype=float)

default_init = np.array([0.6985507, 3.0525788, 1.1474566, 1.8999833, 0.6495420])
default_init = None

def optimize_mixture_model(index, init=None):

    # Select indices and get data.
    indices = finite_indices[npm.query_around_point(kdt, X[index], **kdt_kwds)]

    y = np.array([data[ln][indices] for ln in config["predictor_label_names"]]).T

    if init is None:
        init = npm.get_initialization_point(y)

    opt_kwds = dict(init=init, data=dict(y=y, N=y.shape[0], D=y.shape[1]))
    opt_kwds.update(default_opt_kwds)

    # Do optimization.
    with stan.suppress_output():
        try:
            p_opt = npm._check_params_dict(model.optimizing(**opt_kwds))

        except:
            p_opt = None

    return (index, p_opt, indices)


def sp_swarm(*indices, **kwargs):

    logging.info("Running single processor swarm")

    with tqdm.tqdm(indices, total=len(indices)) as pbar:

        for index in indices:
            if done[index]: continue

            _, result, kdt_indices = optimize_mixture_model(index, default_init)

            pbar.update()

            if result is not None:
                done[index] = True
                results[index] = npm._pack_params(**result)

                if C > 0:
                    nearby_indices = np.atleast_1d(kdt_indices[:C + 1])
                    updated = nearby_indices.size - sum(done[nearby_indices])

                    done[nearby_indices] = True
                    results[nearby_indices] = npm._pack_params(**result)
                    pbar.update(updated)


    return None



def mp_swarm(*indices, max_random_starts=3, in_queue=None, candidate_queue=None,
    out_queue=None):

    def _random_index():
        yield from np.random.choice(indices, max_random_starts, replace=False)

    _ri = _random_index()
    random_start = lambda *_: (_ri.__next__(), default_init)

    swarm = True

    while swarm:

        for func in (in_queue.get_nowait, random_start):
            try:
                index, init = func()

            except mp.queues.Empty:
                logging.info("Using a random index to start")
                continue

            except StopIteration:
                logging.warning("Swarm is bored")
                sleep(5)

            except:
                logging.exception("Unexpected exception:")
                swarm = False

            else:
                if index is None and init is False:
                    swarm = False
                    break

                try:
                    _, result, kdt_indices = optimize_mixture_model(index, init)

                except:
                    logging.exception("Exception when optimizing on {} from {}"\
                        .format(index, init))
                    break

                out_queue.put((index, result))

                if result is not None:
                    if C > 0:
                        # Assign the closest points to have the same result.
                        # (On the other end of the out_qeue we will deal with
                        # multiple results.)
                        out_queue.put((kdt_indices[:C + 1], result))

                    # Candidate next K points
                    K = 3
                    candidate_queue.put((kdt_indices[C + 1:C + 1 + K], result))

            break

    return None


# Set MS as DONE.
#done[data["absolute_rp_mag"] > 2.5] = True

P = 10 # mp.cpu_count()

do_indices = np.where((~done) * finite)[0]

D = do_indices.size
# Save progress
S, save_interval = (0, config.get("save_interval", 100000))
results_path = config.get("results_path", "results.pkl")

logging.info("Intermediate results will be saved to {} every {} steps".format(
    results_path, save_interval))

with mp.Pool(processes=P) as pool:

    manager = mp.Manager()

    in_queue = manager.Queue()
    candidate_queue = manager.Queue()
    out_queue = manager.Queue()

    swarm_kwds = dict(max_random_starts=10000,
                      in_queue=in_queue,
                      out_queue=out_queue,
                      candidate_queue=candidate_queue)


    j = []
    for _ in range(P):
        j.append(pool.apply_async(mp_swarm, do_indices, kwds=swarm_kwds))

    # The swarm will just run at random initial points until we communicate
    # back that the candidates are good.

    with tqdm.tqdm(total=D) as pbar:

        while True:

            has_candidates, has_results = (True, True)

            # Check for candidates.
            try:
                r = candidate_queue.get_nowait()

            except mp.queues.Empty:
                has_candidates = False

            else:
                candidate_indices, init = r
                candidate_indices = np.atleast_1d(candidate_indices)

                for index in candidate_indices:
                    if not done[index] and not queued[index] and finite[index]:
                        in_queue.put((index, init))
                        queued[index] = True


            # Save intermediate results as necessary.
            if int(pbar.n / float(save_interval)) >= S:
                logging.info("Saving intermediate results")
                with open(results_path, "wb") as fp:
                    pickle.dump(results, fp, -1)
                S += 1

            # Check for output.
            try:
                r = out_queue.get(timeout=5)

            except mp.queues.Empty:
                has_results = False

            else:
                index, result = r
                index = np.atleast_1d(index)

                updated = index.size - sum(done[index])
                done[index] = True
                if result is not None:
                    results[index] = npm._pack_params(**result)
                    pbar.update(updated)

            if not has_candidates and not has_results:
                break

# Save intermediate results to disk.
with open(results_path, "wb") as fp:
    pickle.dump(results, fp, -1)

logging.info("Saved intermediate results to {}".format(results_path))


# Clean up difficult cases.
logging.info("Cleaning up any difficult cases.")
while sum(~done * finite) > 0:
    sp_swarm(*np.where((~done) * finite)[0])

with open(results_path, "wb") as fp:
    pickle.dump(results, fp, -1)

logging.info("Results written to {}".format(results_path))
