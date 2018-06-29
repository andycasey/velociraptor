
import pickle
import logging
import numpy as np
import multiprocessing as mp
import tqdm
import os
import yaml
from time import time
from astropy.io import fits

import npm_utils as npm
import stan_utils as stan


with open(npm.CONFIG_PATH, "r") as fp:
    config = yaml.load(fp)


# Load the data.
data = fits.open(config["data_path"])[1].data


indices_kwds = dict(filename=config["indices_path"], dtype=np.int32)
indices_bpr = config["kdtree_maximum_points"] \
            * np.dtype(indices_kwds["dtype"]).itemsize


N, L = (len(data), 4 * len(config["predictor_label_names"]) + 1)        
results_kwds = dict(filename=config["results_path"], dtype=np.float32)
results_bpr = N * np.dtype(results_kwds["dtype"]).itemsize

# Create a results file.
if not os.path.exists(config["results_path"]):
    logging.info("Creating results file: {}".format(config["results_path"]))

    fp = np.memmap(mode="w+", shape=(N, L), **results_kwds)
    fp[:] = np.nan
    fp.flush()
    del fp


def initialize_from_nearby_point(indices):
    """
    Look for an initialization point from nearby results.
    """

    for index in indices:
        fp = np.memmap(mode="c", shape=(L, ), offset=results_bpr * index, 
                       **results_kwds)

        if np.all(np.isfinite(fp)):
            values = npm._unpack_params(np.copy(fp))
            keys = ("theta",
                    "mu_single", "sigma_single",
                    "mu_multiple", "sigma_multiple")
            del fp
            return (index, dict(zip(keys, values)))

        else:
            del fp
            continue

    return None


def optimize_npm_at_point(index, indices, data, config):

    # Get indices and load data.
    y = np.array([data[ln][indices] for ln in config["predictor_label_names"]]).T
    N, D = y.shape

    init_dict = initialize_from_nearby_point(indices)

    if init_dict is None:
        init_from_index = index
        init_dict = npm.get_initialization_point(y)
        
    else:
        init_from_index, init_dict = init_dict

    opt_kwds = dict(
        data=dict(y=y, N=N, D=D),
        init=init_dict,
        verbose=False,
        tol_obj=7./3 - 4./3 - 1, # machine precision
        tol_grad=7./3 - 4./3 - 1, # machine precision
        tol_rel_grad=1e3,
        tol_rel_obj=1e4,
        iter=10000)
    opt_kwds.update(config.get("optimisation_kwds", {}))

    # Make sure that some entries have the right units.
    for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
        if key in opt_kwds:
            opt_kwds[key] = float(opt_kwds[key])

    model = stan.load_stan_model(config["model_path"])

    S = config.get("share_optimised_result_with_nearest", 0)
    
    try:
        p_opt = model.optimizing(**opt_kwds)
    
    except:
        return 0

    result = npm._pack_params(**p_opt)

    # Update result in the memory-mapped array
    fp = np.memmap(mode="r+", shape=(L, ), offset=results_bpr * index,
                   **results_kwds)
    fp[:] = result
    fp.flush()
    del fp

    for nearby_index in indices[:1 + S]:
        if nearby_index == index: continue

        # Load the section in read/write mode.
        fp = np.memmap(mode="r", shape=(L, ), offset=results_bpr * nearby_index,
                       **results_kwds)
        if not np.all(np.isfinite(fp)):
            fp[:] = result
            fp.flush()
        del fp

    return S


def select_next_point(indices):
    for index in indices:
        fp = np.memmap(mode="r", shape=(L, ), offset=results_bpr * index,
                       **results_kwds)
        if not np.all(np.isfinite(fp)):
            del fp
            return index
    return None


# Generator for a random index to run when the swarm gets bored.
def indices():
    yield from np.random.choice(N, N, replace=False)


def run_swarm(queue):
    
    # See if there is anything in the queue.
    try:
        index = queue.get(timeout=0)

    except:
        index = np.random.choice(N, 1)

        
        # Get indices for this index.
        indices = np.memmap(mode="c", 
                            shape=(config["kdtree_maximum_points"], ),
                            offset=index * indices_bpr,
                            **indices_kwds)

        if not any(indices > 0):
            return None

        # Only use positive values.
        indices = indices[indices >= 0]

        S = optimize_npm_at_point(index, indices, data, config)
        index = select_next_point(indices[1 + S:])

        if index is not None:
            queue.put(index)

    return None


q = mp.Queue()
P = mp.cpu_count()
pool = mp.Pool(processes=P)
for _ in tqdm.tqdm(pool.imap_unordered(run_swarm, (q for _ in range(N))), total=N):
    pass

pool.join()
pool.close()


