

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
from scipy.special import logsumexp

import george

import npm_utils as npm
import stan_utils as stan


USE_SV_MASK = True

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
        all_label_names.extend(model_config["kdtree_label_names"])

    all_label_names = list(np.unique(all_label_names))     

    # Mask for finite data points.
    finite = np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0)
    
    USE_SV_MASK = config["sv_mask"]

    if USE_SV_MASK:
        # Mask for science verifiation
        with open("sv.mask", "rb") as fp:
            sv_mask = pickle.load(fp)

        sv_mask = sv_mask[finite]

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

        logging.info(f"Running model {model_name} with config:\n{model_config}")

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

        
        # Optimize the non-parametric model for those sources.
        results = np.zeros((M, 5))
        done = np.zeros(M, dtype=bool)

        def optimize_mixture_model(index, inits=None, scalar=5):

            # Select indices and get data.
            d, nearby_idx, meta = npm.query_around_point(kdt, X[index], **kdt_kwds)

            y = Y[nearby_idx]
            ball = X[nearby_idx]

            if inits is None:
                inits = npm.get_rv_initialisation_points(y, scalar=scalar)

            # Update meta dictionary with things about the data.
            meta = dict(max_log_y=np.log(np.max(y)),
                        N=nearby_idx.size,
                        y_percentiles=np.percentile(y, [16, 50, 84]),
                        ball_ptps=np.ptp(ball, axis=0),
                        ball_medians=np.median(ball, axis=0),
                        init_points=inits,
                        kdt_indices=nearby_idx)

            data_dict = dict(y=np.atleast_2d(y).T,
                             N=y.size,
                             scalar=scalar,
                             D=1,
                             max_log_y=np.log(np.max(y)))
            #for k, v in model_config["parameter_bounds"].items():
            #    data_dict["{}_bounds".format(k)] = v

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

                """
                if sum(done) > 550 and sum(done) < 570:

                    theta, mu_single, sigma_single, mu_multiple, sigma_multiple = npm._pack_params(**p_opt)


                    fig, ax = plt.subplots()
                    xi = np.linspace(0, 20, 1000)

                    y_s = npm.norm_pdf(xi, mu_single, sigma_single, theta)
                    y_m = npm.lognorm_pdf(xi, mu_multiple, sigma_multiple, theta)

                    ax.plot(xi, y_s, c="tab:blue")
                    ax.plot(xi, y_m, c="tab:red")

                    p_single = np.exp(np.log(y_s) - logsumexp([np.log(y_s), np.log(y_m)], axis=0))

                    ax.plot(xi, p_single, c="k")

                    ax.set_title(f"{index}: {theta:.1e} {mu_single:.2f} {sigma_single:.2f} {sigma_multiple:.2f}")

                    ax.hist(y, bins=np.linspace(0, 20, 20), alpha=0.5, facecolor="#666666", normed=True)


                if sum(done) > 570:
                    raise a
                """

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

        # Do not use bad results.

        # Bad results include:
        # - Things that are so clearly discrepant in every parameter.
        # - Things that are on the edge of the boundaries of parameter space.
        sigma = np.abs(results - np.median(results, axis=0)) \
              / np.std(results, axis=0)
        sigma = np.sum(sigma, axis=1)

        tol_sigma, tol_proximity = (10, 1e-2)

        lower_bounds = np.array([0.5, 0.5, 0.05, -np.inf, 0.20])
        upper_bounds = np.array([1.0, 15, 10, +np.inf, 1.6])

        not_ok_bound = np.any(
            (np.abs(results - lower_bounds) <= tol_proximity) \
          + (np.abs(results - upper_bounds) <= tol_proximity), axis=1)

        not_ok_sigma = sigma > tol_sigma

        not_ok = not_ok_bound + not_ok_sigma

        print(f"There were {sum(not_ok_sigma)} results discarded for being outliers")
        print(f"There were {sum(not_ok_bound)} results discarded for being close to the edge")
        print(f"There were {sum(not_ok)} results discarded in total")

        indices = indices[~not_ok]
        results = results[~not_ok]

        # Run the gaussian process on the single star estimates.
        gp_block_size = 10000
        G = 5 # number of kernel hyperparameters
        gp_predict_indices = (0, 1, 2, 3, 4)
        gp_parameters = np.zeros((len(gp_predict_indices), G))
        gp_predictions = np.nan * np.ones((X.shape[0], 2 * len(gp_predict_indices)))

        x = X[indices]

        #randn = np.random.choice(X.shape[0], 50000, replace=False)
            

        for i, index in enumerate(gp_predict_indices):

            y = results[:, index]

            metric = np.var(x, axis=0)
            kernel = george.kernels.Matern32Kernel(metric, ndim=x.shape[1])

            gp = george.GP(kernel, 
                           mean=np.mean(y), fit_mean=True,
                           white_noise=np.log(np.std(y)), fit_white_noise=True)

            assert len(gp.parameter_names) == G

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

            logging.info(f"Predicting {model_name} {index}")

            if USE_SV_MASK:
                p, p_var = gp.predict(y, X[sv_mask], return_var=True)
                gp_predictions[sv_mask, 2*i] = p
                gp_predictions[sv_mask, 2*i+1] = p_var

            else:
                with tqdm.tqdm(total=X.shape[0]) as pb:
                    for b in range(B):
                        s, e = (b * gp_block_size, (b + 1)*gp_block_size)
                        p, p_var = gp.predict(y, X[s:1+e], return_var=True)

                        gp_predictions[s:1+e, 2*i] = p
                        gp_predictions[s:1+e, 2*i + 1] = p_var

                        pb.update(e - s)


            """

            p, p_var = gp.predict(y, X[randn], return_var=True)
            gp_predictions[randn, 2*i] = p
            gp_predictions[randn, 2*i + 1] = p_var
            
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            scat = ax.scatter(X.T[0][randn], X.T[1][randn], 
                c=gp_predictions[:, 2*i][randn], s=1)
            cbar = plt.colorbar(scat)

            ax.set_title(f"{index} mu")

            fig, ax = plt.subplots()
            scat = ax.scatter(X.T[0][randn], X.T[1][randn], 
                c=np.sqrt(gp_predictions[:, 2*i + 1][randn]), s=1)
            cbar = plt.colorbar(scat)

            ax.set_title(f"{index} sigma")
            """

        model_results[model_name] = [indices, results, gp_parameters, gp_predictions]

    # Save the predictions, and the GP hyperparameters.
    save_dict = dict(config=config, models=model_results)
    
    with open(results_path, "wb") as fp:
        pickle.dump(save_dict, fp)

    logging.info(f"Saved output to {results_path}")

    raise a