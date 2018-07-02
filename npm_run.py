
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
C = config.get("share_optimised_result_with_nearest", 0)

kdt, scale, offset = npm.build_kdtree(X[finite], 
    relative_scales=config.get("kdtree_relative_scales", None))

kdt_kwds = dict(offset=offset, scale=scale, full_output=False)
kdt_kwds.update(
    minimum_radius=config.get("kdtree_minimum_radius", None), # DEFAULT
    minimum_points=config.get("kdtree_minimum_points", 1024), # DEFAULT
    maximum_points=config.get("kdtree_maximum_points", 8192)) # DEFAULT

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


done = np.zeros(N, dtype=bool)
queued = np.zeros(N, dtype=bool)
results = np.nan * np.ones((N, L), dtype=float)

indices_bpr = config["kdtree_maximum_points"] * np.dtype(np.int32).itemsize


def optimize_mixture_model(index, init=None):

    # Select indices and get data.
    indices = finite_indices[npm.query_around_point(kdt, X[index], **kdt_kwds)]
    """
    with open(config["indices_path"], "br") as indices_fd:
        indices_fd.seek(indices_bpr * index)
        indices = np.fromfile(indices_fd, dtype=np.int32,
                              count=config["kdtree_maximum_points"])
    """

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



def swarm(*indices, max_random_starts=3, in_queue=None, candidate_queue=None, 
    out_queue=None):

    def _random_index():
        yield from np.random.choice(indices, max_random_starts, replace=False)

    _ri = _random_index()
    random_start = lambda *_: (_ri.__next__(), None)

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

                _, result, kdt_indices = optimize_mixture_model(index, init)

                out_queue.put((index, result))

                if result is not None:
                    if C > 0:
                        # Assign the closest points to have the same result.
                        # (On the other end of the out_qeue we will deal with 
                        # multiple results.)
                        out_queue.put((kdt_indices[:C + 1], result))

                    # Candidate next K points
                    K = 10
                    candidate_queue.put((kdt_indices[C + 1:C + 1 + K], result))

            break

    return None



P = 20 # mp.cpu_count()

with mp.Pool(processes=P) as pool:

    manager = mp.Manager()
    
    in_queue = manager.Queue()
    candidate_queue = manager.Queue()
    out_queue = manager.Queue()

    swarm_kwds = dict(max_random_starts=10,
                      in_queue=in_queue, 
                      out_queue=out_queue,
                      candidate_queue=candidate_queue)

    j = []
    for _ in range(P):
        j.append(pool.apply_async(swarm, finite_indices, kwds=swarm_kwds))

    # The swarm will just run at random initial points until we communicate
    # back that the candidates are good.

    with tqdm.tqdm(total=F) as pbar:

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

            # Add indices to the queue that have not been done yet so that the
            # swarm does not get bored and start using random indices?

            #print("approximate queue sizes: {} {} {}".format(
            #    in_queue.qsize(), candidate_queue.qsize(), out_queue.qsize()))

            # If the number of items in the in_queue reaches less than 50%,
            # and we have already processed 50% of the sources,
            # then we should be adding items,... right?

            # Or we just do a cleean up afterwards.


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


# Clean up any difficult cases.
with mp.Pool(processes=P) as pool:

    manager = mp.Manager()
    
    in_queue = manager.Queue()
    candidate_queue = manager.Queue()
    out_queue = manager.Queue()

    swarm_kwds = dict(max_random_starts=0,
                      in_queue=in_queue, 
                      out_queue=out_queue,
                      candidate_queue=candidate_queue)

    # Do a check for entries that are not done.
    not_done = np.where((~done * finite))[0]
    ND = not_done.size

    logging.info("{} not done".format(ND))
    for index in not_done:

        # Get nearest points that are done.
        with open(config["indices_path"], "br") as indices_fd:
            indices_fd.seek(indices_bpr * index)
            indices = np.fromfile(indices_fd, dtype=np.int32,
                                  count=config["kdtree_maximum_points"])

        in_queue.put((index, results[indices[done[indices]][0]]))

    j = []
    for _ in range(P):
        j.append(pool.apply_async(swarm, finite_indices, kwds=swarm_kwds))


    with tqdm.tqdm(total=ND) as pbar:

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
                sleep(1)
                # Do a check for entries that are not done.
                not_done = np.where((~done * finite))[0]
                ND = not_done.size

                if ND == 0:
                    break


results_path = config.get("results_path", "results.pkl")
with open(results_path, "wb") as fp:
    pickle.dump(results, fp, -1)

logging.info("Results written to {}".format(results_path))