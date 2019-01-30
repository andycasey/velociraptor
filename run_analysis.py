

"""
Analysis script for the Velociraptor project.
"""

import logging
import multiprocessing as mp
import numpy as np
import os
import pickle
import sys
import tqdm
import yaml
from time import (sleep, time)
from astropy.io import fits
from scipy import optimize as op

import george

import npm_utils as npm
import stan_utils as stan


if __name__ == "__main__":

    config_path = sys.argv[1]

    with open(config_path, "r") as fp:
        config = yaml.load(fp)

    random_seed = int(config["random_seed"])
    np.random.seed(random_seed)

    logging.info(f"Config path: {config_path} with seed {random_seed}")

    # Check results path now so we don't die later.
    results_path = config["results_path"]

    # Load data.
    data = fits.open(config["data_path"])[1].data

    # Get a list of all relevant label names
    all_label_names = []
    for model_name, model_config in config["models"].items():
        all_label_names.append(model_config["predictor_label_name"])
        all_label_names.extend(model_config["kdt_label_names"])

    all_label_names = list(np.unique(all_label_names))     

    # Mask for finite data points.
    finite = np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0)


    # Load the model.
    model = stan.load_stan_model(config["model_path"], verbose=False)

    # Make sure that some entries have the right type.
    default_opt_kwds = config.get("optimisation_kwds", {})
    for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
        if key in default_opt_kwds:
            default_opt_kwds[key] = float(default_opt_kwds[key])

    logging.info("Optimization keywords: {}".format(default_opt_kwds))

    M = config["number_of_sources"]
    indices = np.random.choice(sum(finite), M, replace=False)

    model_results = dict()
    for model_name, model_config in config["models"].items():

        # Set up a KD-tree.
        X = np.vstack([data[ln][finite] for ln in model_config["kdtree_label_names"]]).T
        Y = np.array(data[model_config["predictor_label_name"]])[finite]

        N, D = X.shape

        kdt, scales, offsets = npm.build_kdtree(X, 
                relative_scales=model_config["kdtree_relative_scales"])

        kdt_kwds = dict(offsets=offsets, scales=scales, full_output=True)
        kdt_kwds.update(
            minimum_radius=model_config["kdtree_minimum_radius"],
            maximum_radius=model_config.get("kdtree_maximum_radius", None),
            minimum_points=model_config["kdtree_minimum_points"],
            maximum_points=model_config["kdtree_maximum_points"],
            minimum_density=model_config.get("kdtree_minimum_density", None))

        logging.info("k-d tree keywords: {}".format(kdt_kwds))

        # Optimize the non-parametric model for those sources.
        results = np.zeros((M, 5))
        done = np.zeros(M, dtype=bool)

        def optimize_mixture_model(index, inits=None):

            # Select indices and get data.
            d, nearby_idx, meta = npm.query_around_point(kdt, X[index], **kdt_kwds)

            y = Y[nearby_idx]
            ball = X[nearby_idx]

            if inits is None:
                inits = npm.get_rv_initialisation_points(y)

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
            for k, v in model_config["parameter_bounds"].items():
                data_dict["{}_bounds".format(k)] = v

            p_opts = []
            ln_probs = []
            for j, init_dict in enumerate(inits):

                opt_kwds = dict(
                    init=init_dict,
                    data=data_dict)
                opt_kwds.update(default_opt_kwds)

                # Do optimization.
                # TODO: Suppressing output is always dangerous.
                with stan.suppress_output(config.get("suppress_stan_output", True)) as sm:
                    try:
                        p_opt = model.optimizing(**opt_kwds)

                    except:
                        logging.exception(f"Exception occurred when optimizing index {index}"\
                                          f" from {init_dict}:")

                    else:
                        if p_opt is not None:
                            p_opts.append(p_opt)
                            ln_probs.append(npm.ln_prob(y, 1, *npm._pack_params(**p_opt)))

                if p_opt is None:
                    stdout, stderr = sm.outputs
                    logging.warning(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")

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



        def mp_swarm(*mp_indices, in_queue=None, out_queue=None, seed=None):

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

                swarm_kwds = dict(in_queue=in_queue,
                                  out_queue=out_queue)


                logging.info("Dumping everything into the queue!")
                for j, index in enumerate(indices):
                    in_queue.put((j, index, ))

                j = []
                for _ in range(P):
                    j.append(pool.apply_async(mp_swarm, [], kwds=swarm_kwds))


                with tqdm.tqdm(total=M) as pbar:

                    while not np.all(done):

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


        # Run the gaussian process on the single star estimates.
        gp_block_size = 10000
        
        gp_predict_indices = (1, 2, 4)
        gp_parameters = np.zeros((len(gp_predict_indices), 4))
        gp_predictions = np.nan * np.ones((X.shape[0], 2 * len(gp_predict_indices)))

        x = X[indices]

        for i, index in enumerate(gp_predict_indices):

            y = results[:, index]

            metric = np.var(x, axis=0)
            kernel = george.kernels.Matern32Kernel(metric, ndim=x.shape[1])

            gp = george.GP(kernel, mean=np.mean(y), fit_mean=True, fit_white_noise=True)


            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25

            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y, quiet=True)

            gp.compute(x)
            logging.info("Initial \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
            logging.info("initial \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

            p0 = gp.get_parameter_vector()

            t_init = time()
            result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
            t_opt = time() - t_init


            gp.set_parameter_vector(result.x)
            logging.info("Result: {}".format(result))
            logging.info("Final logL = {:.2f}".format(gp.log_likelihood(y)))
            logging.info("Took {:.0f} seconds to optimize".format(t_opt))

            gp_parameters[i] = result.x

            # Predict the quantity and the variance.
            B = int(np.ceil(X.shape[0] / gp_block_size))

            for b in tqdm.tqdm(range(B)):
                s, e = (b * gp_block_size, (b + 1)*gp_block_size)
                p, p_var = gp.predict(y, X[s:1+e], return_var=True)

                gp_predictions[s:1+e, 2*i] = p
                gp_predictions[s:1+e, 2*i + 1] = p_var

        model_results[model_name] = [results, gp_parameters, gp_predictions]

    # Save the predictions, and the GP hyperparameters.
    save_dict = dict(config=config, results=model_results)
    
    with open(results_path, "wb") as fp:
        pickle.dump(save_dict, fp)

    logging.info(f"Saved output to {results_path}")
    

"""

raise a

# Do this for some validation set?

pred, pred_var = gp.predict(y, x, return_var=True)

kwds = dict(s=1)

fig, axes = plt.subplots(2, 3, figsize=(12, 4))

# Data.model
axes[0, 0].scatter(x.T[0], x.T[1], c=y, **kwds)
axes[1, 0].scatter(x.T[0], x.T[2], c=y, **kwds)

# Model.
axes[0, 1].scatter(x.T[0], x.T[1], c=pred, **kwds)
axes[1, 1].scatter(x.T[0], x.T[2], c=pred, **kwds)

# Residual
residuals = pred - y
vminmax = np.percentile(np.abs(residuals), 95)

residual_kwds = kwds.copy()
residual_kwds.update(vmin=-vminmax, vmax=+vminmax)
scat = axes[0, 2].scatter(x.T[0], x.T[1], c=pred-y, **residual_kwds)
axes[1, 2].scatter(x.T[0], x.T[1], c=pred-y, **residual_kwds)


#cbar = plt.colorbar(scat)


for ax_col, ylabel in zip(axes, ("absolute rp mag", "apparent rp mag")):
    # Common limits for each column.
    ylims = np.array([ax.get_ylim() for ax in ax_col])
    ylims = (np.max(ylims), np.min(ylims))

    for ax in ax_col:
        #ax.set_ylabel(r"$\textrm{{{0}}}$".format(ylabel))
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylims)
        #ax.set_xlabel(r"$\textrm{{bp - rp}}$")
        ax.set_xlabel("bp - rp")
        ax.set_ylim(ylims)


for ax, title in zip(axes[0], ("data", "model", "residual")):
    #x.set_title(r"$\textrm{{{0}}}$".format(title))
    ax.set_title(title)



def lnprob(p):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()


initial = gp.get_parameter_vector()
ndim, nwalkers = len(initial), 32
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
threads = 1 if config.get("multiprocessing", False) else mp.cpu_count()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads)


burn = 500

print("Running burn-in...")
for result in tqdm.tqdm(sampler.sample(p0, iterations=burn), total=burn):
    continue

sampler.reset()

prod = 1000
print("Running production...")
for result in tqdm.tqdm(sampler.sample(p0, iterations=prod), total=prod):
    continue

fig, axes = plt.subplots(ndim, 1)

for j, ax in enumerate(axes):

    for k in range(nwalkers):
        ax.plot(sampler.chain[k, :, j], c="k", alpha=0.1)


# Plot 50 draws for one star.
N_draws = 50
results = np.zeros((M, N_draws))
for i in range(N_draws):

    w = np.random.randint(sampler.chain.shape[0])
    n = np.random.randint(sampler.chain.shape[1])

    gp.set_parameter_vector(sampler.chain[w, n])

    results[:, i] = gp.sample_conditional(y, x)



time = time()
print("Running production...")
sampler.run_mcmc(p0, 1000);
t_prod = time() - time
print(f"Production took {t_prod:.0f} seconds")
"""