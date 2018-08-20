
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

with open("the-battery-stars.rv.yaml", "r") as fp:
    config = yaml.load(fp)

config["results_path"] = "results/the-battery-stars-comparison.rv_single_epoch_scatter.pkl"


with open("the-battery-stars.astrometry.yaml", "r") as fp:
    config = yaml.load(fp)
config["results_path"] = "results/the-battery-stars-comparison.astrometric_unit_weight_error.pkl"

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
L = 4 * len(config["predictor_label_names"]) + 1 # + 1 if using mu_multiple_uv
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

default_opt_kwds = config.get("optimisation_kwds", {})

# Make sure that some entries have the right units.
for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
    if key in default_opt_kwds:
        default_opt_kwds[key] = float(default_opt_kwds[key])

logging.info("k-d tree keywords: {}".format(kdt_kwds))
logging.info("optimization keywords: {}".format(default_opt_kwds))


done = np.zeros(N, dtype=bool)
queued = np.zeros(N, dtype=bool)
results = np.nan * np.ones((N, L), dtype=float)

default_init = dict(zip(
    ("theta", "mu_single", "sigma_single", "mu_multiple", "sigma_multiple"),
    np.array([0.75, 1, 0.5, 1, 0.75])))
default_init["mu_multiple_uv"] = 0.1
default_init = npm._check_params_dict(default_init)
bounds = config["parameter_bounds"]

def optimize_mixture_model(index, init=None):

    # Select indices and get data.
    indices = finite_indices[npm.query_around_point(kdt, X[index], **kdt_kwds)]

    y = np.array([data[ln][indices] for ln in config["predictor_label_names"]]).T

    if init is None:
        init = npm.get_initialization_point(y)

    data_dict = dict(y=y, 
                     N=y.shape[0], 
                     D=y.shape[1],
                     max_log_y=np.log(np.max(y)))
    for k, v in bounds.items():
        data_dict["{}_bounds".format(k)] = v
    
    trial_results = []
    for j, init_dict in enumerate((init, npm.get_initialization_point(y), "random")):

        # CHeck that the parameters are bounded?
        if isinstance(init_dict, dict):
            if bounds is not None:
                for k, (lower, upper) in bounds.items():
                    if not (upper > init_dict[k] > lower):
                        logging.info("Clipping initial value of {} from {} to within ({}, {})".format(
                            k, init_dict[k], lower, upper))
                        offset = 0.01 * (upper - lower)
                        init_dict[k] = np.clip(init_dict[k], lower + offset, upper - offset)


        opt_kwds = dict(
            init=init_dict, 
            data=data_dict)
        opt_kwds.update(default_opt_kwds)

        # Do optimization.
        with stan.suppress_output() as sm:
            try:
                p_opt = model.optimizing(**opt_kwds)

            except:
                p_opt = None
                # TODO: Consider relaxing the optimization tolerances!

        if p_opt is None:
            # Capture stdout and stderr so we can read it later.
            stdout, stderr = sm.stdout, sm.stderr

        trial_results.append(p_opt)

        if p_opt is None:
            logging.warning("Exception when optimizing on index {} from "\
                            "initial point {}:".format(index, init_dict))
            logging.warning(stdout)
            logging.warning(stderr)
            raise

        else:

            tolerance = 1e-2
            for k, v in p_opt.items():
                if k not in bounds \
                or (k == "theta" and (v >= 1-tolerance)): continue

                lower, upper = bounds[k]
                if np.abs(v - lower) <= tolerance \
                or np.abs(v - upper) <= tolerance:
                    logging.warning("Optimised {} at edge of grid ({} < {} < {})"
                                    " - ignoring".format(k, lower, v, upper))
                    break

            else:
                break


    else:
        # TODO: Consider relaxing optimization tolerances!
        logging.warning("Optimization did not converge from any initial point "\
                        "trialled. Consider relaxing optimization tolerances!")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(y, bins=100, facecolor="#cccccc", normed=True)

        # Plot a model at some optimization point.
        init = npm.get_initialization_point(y)
        N, D = y.shape

        xi = np.linspace(0, max(y), 1000)

        ax.plot(xi, npm.norm_pdf(xi, init["mu_single"], init["sigma_single"],
            init["theta"]), c="r")
        ax.plot(xi, npm.lognorm_pdf(xi, init["mu_multiple"], init["sigma_multiple"],
            init["theta"]), c="r", linestyle=":")
        

        ax.plot(xi, npm.norm_pdf(xi, default_init["mu_single"], default_init["sigma_single"],
            default_init["theta"]), c="b")
        ax.plot(xi, npm.lognorm_pdf(xi, default_init["mu_multiple"], default_init["sigma_multiple"],
            default_init["theta"]), c="b", linestyle=":")
        

        raise a

    p_opt = npm._check_params_dict(p_opt)

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
                #logging.info("Using a random index to start")
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
                    K = 0
                    #candidate_queue.put((kdt_indices[C + 1:C + 1 + K], result))

            break

    return None




with open("data/the-battery-stars.indices.pkl", "rb") as fp:
    battery_star_indices = pickle.load(fp)

np.random.seed(42)
comp_indices_path = "data/the-battery-stars-comparison.indices.pkl"
if os.path.exists(comp_indices_path):
    with open(comp_indices_path, "rb") as fp:
        comp_indices = pickle.load(fp)

else:
    # Only take those with finite values.
    fin = np.all(np.isfinite(X[battery_star_indices]), axis=1)
    finite_battery_star_indices = battery_star_indices[fin]


    comp_indices = np.zeros_like(finite_battery_star_indices)

    for i, index in enumerate(finite_battery_star_indices):
        kdt_indices = finite_indices[npm.query_around_point(kdt, X[index], **kdt_kwds)]

        for each in kdt_indices:
            if each not in finite_battery_star_indices:
                comp_indices[i] = each
                break
        else:
            raise a

    with open(comp_indices_path, "wb") as fp:
        pickle.dump(comp_indices, fp)





sp_swarm(*comp_indices)


# Save results.
results_path = config.get("results_path", "results.pkl")

with open(results_path, "wb") as fp:
    pickle.dump(results, fp, -1)

logging.info("Saved results to {}".format(results_path))




with open("results/the-battery-stars-comparison.rv_single_epoch_scatter.pkl", "rb") as fp:
    rv_results = pickle.load(fp)

with open("results/the-battery-stars-comparison.astrometric_unit_weight_error.pkl", "rb") as fp:
    astrometric_results = pickle.load(fp)

from astropy.table import Table
subset = Table(data[comp_indices])

keys = ("theta", "mu_single", "sigma_single", "mu_multi", "sigma_multi")

for i, key in enumerate(keys):
    subset["rv_{}".format(key)] = rv_results[comp_indices, i]
    subset["astrometric_{}".format(key)] = astrometric_results[comp_indices, i]

finite = np.isfinite(subset["astrometric_theta"] * subset["rv_theta"])
subset = subset[finite]
print(len(subset))

subset.write("results/the-battery-stars-comparison.results.fits", overwrite=True)
