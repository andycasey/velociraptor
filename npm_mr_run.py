
import pickle
import logging
import numpy as np
import multiprocessing as mp
import tqdm
import os
import io
import gc
import sys
import yaml
import warnings
import resource
from time import time, sleep
from astropy.io import fits

import npm_utils as npm
import stan_utils as stan


VERBOSE = False

logging.info("RLIMIT_NOFILE = {}".format(resource.getrlimit(resource.RLIMIT_NOFILE)))

try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, -1))

except ValueError:
    logging.exception("Could not update RLIMIT_NOFILE")

warnings.filterwarnings("ignore")

with open(npm.CONFIG_PATH, "r") as config_fp:
    config = yaml.load(config_fp)


# Load the data.
with fits.open(config["data_path"]) as image:
    data = image[1].data

    all_label_names = list(config["kdtree_label_names"]) \
                    + list(config["predictor_label_names"])

    # Set up a KD-tree.
    X = np.vstack([data[ln] for ln in config["kdtree_label_names"]]).T
    finite = np.where(np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0))[0]

N, _ = X.shape
F = finite.size
L = 4 * len(config["predictor_label_names"]) + 1

indices_kwds = dict(filename=config["indices_path"], dtype=np.int32)
indices_bpr = config["kdtree_maximum_points"] \
            * np.dtype(indices_kwds["dtype"]).itemsize

results_kwds = dict(filename=config["results_path"], dtype=np.float32)
results_bpr = L * np.dtype(results_kwds["dtype"]).itemsize

# Create a results file.
if not os.path.exists(results_kwds["filename"]):
    logging.info("Creating results file: {}".format(results_kwds["filename"]))
    fp = np.memmap(mode="w+", shape=(N, L), **results_kwds)
    del fp
    
            

def initialize_from_nearby_point(indices, ignore_index):
    """
    Look for an initialization point from nearby results.
    """

    for index in indices:
        if index == ignore_index: continue
        print("Opening at index {}".format(index))

        #fp = np.memmap(results_fd, mode="r", shape=(L, ), 
        #               offset=results_bpr * index, dtype=np.float32)

        with open(results_kwds["filename"], "br+") as results_fd:
            results_fd.seek(results_bpr * index)
            fp = np.fromfile(results_fd, dtype=np.float32, count=L)

        if any(fp > 0):
            keys = ("theta",
                    "mu_single", "sigma_single",
                    "mu_multiple", "sigma_multiple")
            init_dict = dict(zip(keys, npm._unpack_params(fp)))
            del fp
            
            return (index, init_dict)

        del fp
        
            
    
    return None


def optimize_npm_at_point(index, indices, model, config, default_opt_kwds,
                          queue):

    print("Opening for optimising at index {}".format(index))

    #fp = np.memmap(results_fd, mode="r+", shape=(L, ), 
    #               offset=results_bpr * index, dtype=np.float32)

    with open(results_kwds["filename"], "br+") as results_fd:
        results_fd.seek(results_bpr * index)
        fp = np.fromfile(results_fd, dtype=np.float32, count=L)


    if any(fp > 0):
        # Already a result for this source.
        print("Already a result for {}".format(index))
        del fp

        return 0

    # Get indices and load data.
    with fits.open(config["data_path"]) as image:
        data = image[1].data
        y = np.array([data[ln][indices] for ln in config["predictor_label_names"]]).T
    
    N, D = y.shape

    init_dict = initialize_from_nearby_point(indices, index)

    if init_dict is None:
        init_from_index = index
        init_dict = npm.get_initialization_point(y)
        
    else:
        init_from_index, init_dict = init_dict

    opt_kwds = dict(init=init_dict, data=dict(y=y, N=N, D=D))
    opt_kwds.update(default_opt_kwds)
    

    S = config.get("share_optimised_result_with_nearest", 0)
    
    try:    
        with stan.suppress_output():
            p_opt = model.optimizing(**opt_kwds)
        
    except:
        del fp
        return 0

    result = npm._pack_params(**p_opt)

    fp = np.memmap(results_kwds["filename"], mode="r+", shape=(L, ),
                   dtype=np.float32, offset=results_bpr * index)
    fp[:] = result
    fp.flush()
    del fp
    
    if S > 0:
        raise a
        for nearby_index in indices[:1 + S]:
            if nearby_index == index: continue

            # Load the section in read/write mode.
            print("Opening to share result at {}".format(nearby_index))
        
            fp = np.memmap(results_fd, mode="r+", shape=(L, ), dtype=np.float32,
                           offset=results_bpr * nearby_index)

            if not any(fp > 0):
                fp[:] = result
                queue.put(1)
            del fp
            
        
    return S


def select_next_point(indices, ignore_index):
    #print("selecting next point {} {}".format(indices, ignore_index))

    for index in indices:
        if index == ignore_index: continue
        print("Selecting nearby result at {}".format(index))

        #fp = np.memmap(results_fd, mode="r", shape=(L, ), dtype=np.float32,
        #               offset=results_bpr * index)

        with open(results_kwds["filename"], "br+") as results_fd:
            results_fd.seek(results_bpr * index)
            fp = np.fromfile(results_fd, dtype=np.float32, count=L)

        if not any(fp > 0):
            del fp
            return index

        del fp
        
    
    return None



def run_swarm(queue, swarm_indices):
    
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

    for index in swarm_indices:
        
        while True:

            # Get indices for this index.
            print("Getting indices for {}".format(index))

            with open(indices_kwds["filename"], "br") as indices_fd:
                indices_fd.seek(indices_bpr * index)
                indices = np.fromfile(indices_fd, dtype=np.int32,
                                      count=config["kdtree_maximum_points"])

            if not any(indices > 0):
                queue.put(1)
                del indices
                break

            # Only use positive values.
            use_indices = indices[indices >= 0]
            del indices

            S = optimize_npm_at_point(index, use_indices, model, config, 
                                      default_opt_kwds, queue)
            
            #index = select_next_point(use_indices[1 + S:], ignore_index=index)
            queue.put(1)
            
            break

            if index is None:
                break
            


manager = mp.Manager()
queue = manager.Queue()

run_swarm(queue, np.random.choice(finite, F, replace=False))

P = mp.cpu_count()
F = finite.size


with mp.Pool(processes=P) as pool:

    # Set swarms going on each thread.
    jobs = []
    for _ in range(P):
        jobs.append(pool.apply_async(
            run_swarm, args=(queue, np.random.choice(finite, F, replace=False, ))))

    # Collect the results from the workers so that we get an accurate idea of
    # the progress.
    try:
        with tqdm.tqdm(total=F) as pbar:
            for _ in range(F):
                pbar.update(queue.get(timeout=30))

    except:
        logging.exception("Failed progressbar")

    # Check that things did not fail in some weird way.
    for job in jobs:
        job.get()
