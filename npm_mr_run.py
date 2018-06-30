
import pickle
import logging
import numpy as np
import multiprocessing as mp
import tqdm
import os
import contextlib
import io
import sys
import yaml
import warnings
from time import time
from astropy.io import fits

import npm_utils as npm
import stan_utils as stan

np.random.seed(42)

warnings.filterwarnings("ignore")

with open(npm.CONFIG_PATH, "r") as fp:
    config = yaml.load(fp)


# Load the data.
data = fits.open(config["data_path"])[1].data

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

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def initialize_from_nearby_point(indices):
    """
    Look for an initialization point from nearby results.
    """

    for index in indices:
        fp = np.memmap(mode="c", shape=(L, ), offset=results_bpr * index, 
                       **results_kwds)

        if any(fp > 0):
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

    fp = np.memmap(mode="r+", shape=(L, ), offset=results_bpr * index,
                   **results_kwds)

    if any(fp > 0):
        del fp
        return 0

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

    model = stan.load_stan_model(config["model_path"], verbose=False)

    S = config.get("share_optimised_result_with_nearest", 0)
    
    try:
        #with open(os.devnull, "w") as devnull:
        #    with contextlib.redirect_stdout(devnull):
        with suppress_stdout_stderr():
            p_opt = model.optimizing(**opt_kwds)
    
    except:
        return 0

    result = npm._pack_params(**p_opt)

    # Update result in the memory-mapped array
    fp[:] = result
    fp.flush()
    del fp

    for nearby_index in indices[:1 + S]:
        if nearby_index == index: continue

        # Load the section in read/write mode.
        fp = np.memmap(mode="r+", shape=(L, ), offset=results_bpr * nearby_index,
                       **results_kwds)
        if not any(fp > 0):
            fp[:] = result
            fp.flush()
        del fp

    return S


def select_next_point(indices):
    for index in indices:
        fp = np.memmap(mode="r", shape=(L, ), offset=results_bpr * index,
                       **results_kwds)
        if not any(fp > 0):
            del fp
            return index
    return None


# Generator for a random index to run when the swarm gets bored.
def indices():
    yield from np.random.choice(N, N, replace=False)


def run_swarm(queue, index=None):
    
    # See if there is anything in the queue.
    if index is None:
        try:
            index = queue.get(timeout=0)
            
        except:
            index = np.random.choice(N, 1)[0]

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


P = mp.cpu_count()
pool = mp.Pool(processes=50)
manager = mp.Manager()
queue = manager.Queue()

for index in tqdm.tqdm(range(N), total=N):
    pool.apply_async(run_swarm, (queue, )).get()

# Do a clean-up check to ensure we actually did every source.

pool.join()
pool.close()


