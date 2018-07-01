
""" Set up and run the non-parametric model. """

import numpy as np
import os
import multiprocessing as mp
import yaml
from astropy.io import fits
import tqdm

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
results = np.nan * np.ones((N, L), dtype=float)



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



def swarm(*indices, queue=None):

    init = [None]
    for index in indices:

        while True:
            _, result, kdt_indices = optimize_mixture_model(index, init.pop(0))

            if queue is not None:
                queue.put(1)

            # Check to see which kdt_indices have not been done.
            done[index] = True

            if result is None:
                # We will say this point is 'done', even though result failed.
                break

            results[index] = npm._pack_params(**result)

            if C > 0:
                # Assign the closest points to have the same result.
                B = sum(done[kdt_indices[:C + 1]])
                done[kdt_indices[:C + 1]] = True
                results[kdt_indices[:C + 1]] = results[index]

                if queue is not None:
                    queue.put(C + 1 - B)

            next_indices = kdt_indices[(~done * finite)[kdt_indices]]
            if len(next_indices) > 0:
                # Initialize the next point from this optimized result.
                index = next_indices[0]
                init.append(result)

            else:
                # All points in the ball were done.
                init.append(None)
                break

    return None



P = 8 # mp.cpu_count()

with mp.Pool(processes=P) as pool:

    queue = mp.Manager().Queue()

    j = []
    for _ in range(P):
        j.append(pool.apply_async(swarm,
                                  np.random.choice(finite_indices, F, False),
                                  kwds=dict(queue=queue)))

    # Collect the results from the workers so that we get an accurate idea of
    # the progress.
    with tqdm.tqdm(total=F) as pbar:
        for _ in range(F):
            pbar.update(queue.get(timeout=30))
