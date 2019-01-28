

"""
Analysis script for the Velociraptor project.
"""

import numpy as np
import multiprocessing as mp
import yaml
import tqdm
import logging
import os
from time import sleep
from astropy.io import fits

import npm_utils as npm
import stan_utils as stan

seed = 42

np.random.seed(seed)

# Load config.
with open("config.rv.yml", "r") as fp:
    config = yaml.load(fp)

# Load data.
data = fits.open(config["data_path"])[1].data

#data = data[:10000]

all_label_names = list(config["kdtree_label_names"]) \
                + list(config["require_finite_label_names"]) \
                + list(config["predictor_label_names"])
all_label_names = list(np.unique(all_label_names))     

# Set up a KD-tree.
finite = np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0)
X = np.vstack([data[ln][finite] for ln in config["kdtree_label_names"]]).T
Y = np.array([data[ln][finite] for ln in config["predictor_label_names"]]).T

N, D = X.shape
C = config["share_optimised_result_with_nearest"]

kdt, scales, offsets = npm.build_kdtree(X, 
        relative_scales=config["kdtree_relative_scales"])

kdt_kwds = dict(offsets=offsets, scales=scales, full_output=True)
kdt_kwds.update(
    minimum_radius=config["kdtree_minimum_radius"],
    maximum_radius=config.get("kdtree_maximum_radius", None),
    minimum_points=config["kdtree_minimum_points"],
    maximum_points=config["kdtree_maximum_points"],
    minimum_density=config.get("kdtree_minimum_density", None))

logging.info("k-d tree keywords: {}".format(kdt_kwds))


# Load the model.
model = stan.load_stan_model(config["model_path"], verbose=False)

default_opt_kwds = config.get("optimisation_kwds", {})

# Make sure that some entries have the right type.
for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
    if key in default_opt_kwds:
        default_opt_kwds[key] = float(default_opt_kwds[key])

logging.info("optimization keywords: {}".format(default_opt_kwds))


# Select the points to run the model on.
M = 10000
indices = np.random.choice(N, M, replace=False)

# Optimize the non-parametric model for those sources.
results = np.zeros((M, 5))
done = np.zeros(M, dtype=bool)
queued = np.zeros(M, dtype=bool)


def optimize_mixture_model(index, inits=None):

    # Select indices and get data.
    d, nearby_idx, meta = npm.query_around_point(kdt, X[index], **kdt_kwds)

    y = Y[nearby_idx]
    ball = X[nearby_idx]

    if inits is None:
        inits = npm.get_rv_initialisation_points(y)

    assert np.all(np.ptp(ball, axis=0) <= 2*np.array(config["kdtree_maximum_radius"]))

    # Update meta dictionary with things about the data.
    meta = dict(max_log_y=np.log(np.max(y)),
                N=nearby_idx.size,
                y_percentiles=np.percentile(y, [16, 50, 84]),
                ball_ptps=np.ptp(ball, axis=0),
                ball_medians=np.median(ball, axis=0),
                init_points=inits,
                kdt_indices=nearby_idx)

    data_dict = dict(y=y,
                     N=y.shape[0],
                     D=y.shape[1],
                     max_log_y=np.log(np.max(y)))
    for k, v in config["parameter_bounds"].items():
        data_dict["{}_bounds".format(k)] = v

    p_opts = []
    ln_probs = []
    for j, init_dict in enumerate(inits):

        opt_kwds = dict(
            init=init_dict,
            data=data_dict)
        opt_kwds.update(default_opt_kwds)

        # Do optimization.
        try:
            p_opt = model.optimizing(**opt_kwds)

        except:
            logging.exception(f"Exception occurred when optimizing index {index}"\
                              f" from {init_dict}:")
            continue

        else:
            if p_opt is not None:
                p_opts.append(p_opt)
                ln_probs.append(npm.ln_prob(y, 1, *npm._pack_params(**p_opt)))

    if len(p_opts) < 1:
        logging.warning("Optimization on index {} did not converge from any "\
                        "initial point trialled. Consider relaxing the "\
                        "optimization tolerances! If this occurs regularly "\
                        "then something is very wrong!".format(index))

        return (index, None, meta)

    # evaluate best.
    else:
        idx = np.argmax(ln_probs)
        p_opt = p_opts[idx]
        meta["init_idx"] = idx

        return (index, p_opt, meta)




def sp_swarm(*sp_indices, **kwargs):

    logging.info("Running single processor swarm")

    with tqdm.tqdm(sp_indices, total=len(sp_indices)) as pbar:

        for j, index in enumerate(sp_indices):
            if done[j]: continue

            _, result, meta = optimize_mixture_model(index)

            pbar.update()

            done[j] = True
            
            if result is not None:
                results[j] = npm._pack_params(**result)
                 
    return None



def mp_swarm(*mp_indices, max_random_starts=3, in_queue=None, candidate_queue=None,
             out_queue=None, seed=None):

    np.random.seed(seed)

    swarm = True

    while swarm:

        try:
            j, index = in_queue.get_nowait()

        except mp.queues.Empty:
            logging.info("Queue is empty")
            break

        except StopIteration:
            logging.warning("Swarm is bored")
            break

        except:
            logging.exception("Unexpected exception:")
            break

        else:
            if index is None and init is False:
                swarm = False
                break

            try:
                _, result, meta = optimize_mixture_model(index)

            except:
                logging.exception(f"Exception when optimizing on {index}")
                out_queue.put((j, index, None, dict()))
            
            else:
                out_queue.put((j, index, result, meta))

    return None



if not config.get("multiprocessing", False):
    sp_swarm(*indices)

else:
    P = mp.cpu_count()

    with mp.Pool(processes=P) as pool:

        manager = mp.Manager()

        in_queue = manager.Queue()
        out_queue = manager.Queue()

        swarm_kwds = dict(max_random_starts=M,
                          in_queue=in_queue,
                          out_queue=out_queue)


        logging.info("Dumping everything into the queue!")
        for j, index in enumerate(indices):
            in_queue.put((j, index, ))

        j = []
        for _ in range(P):
            j.append(pool.apply_async(mp_swarm, [], kwds=swarm_kwds))


        with tqdm.tqdm(total=M) as pbar:

            while True:

                # Check for output.
                try:
                    r = out_queue.get(timeout=30)

                except mp.queues.Empty:
                    logging.info("No results")
                    break

                else:
                    j, index, result, meta = r

                    done[j] = True
                    if result is not None:
                        results[j] = npm._pack_params(**result)

                    pbar.update(1)



# Run the gaussian process on the estimates.

# Predict the properties of single stars.

raise a