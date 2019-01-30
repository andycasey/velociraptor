
"""
Run this after run_analysis.py
"""

import logging
import numpy as np
import sys
import pickle
import tqdm
from collections import OrderedDict
from astropy.io import fits
from astropy.table import Table
from scipy.special import logsumexp
from scipy import stats


import npm_utils as npm


def lnprob(y, theta, s_mu, s_sigma, b_mu, b_sigma):

    s_ivar = np.abs(s_sigma)**-2
    b_ivar = np.abs(b_sigma)**-2
    hl2p = 0.5 * np.log(2*np.pi)
    
    s_lpdf = -hl2p + 0.5 * np.log(s_ivar) \
           - 0.5 * (y - s_mu)**2 * s_ivar
    
    b_lpdf = -np.log(y*b_sigma) - hl2p \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar
    b_lpdf[~np.isfinite(b_lpdf)] = -np.inf

    return np.vstack([s_lpdf, b_lpdf]).T + np.log([theta, 1-theta])



if __name__ == "__main__":

    result_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(result_path, "rb") as fp:
        results = pickle.load(fp)

    config = results["config"]
    data = fits.open(config["data_path"])[1].data

    all_label_names = []
    for model_name, model_config in config["models"].items():

        all_label_names.append(model_config["predictor_label_name"])
        all_label_names.extend(model_config["kdtree_label_names"])

    all_label_names = list(np.unique(all_label_names))

    finite = np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0)

    K = config.get("number_of_draws", 100)
    model_names = config["models"]
    M = len(model_names)
    MJ = M + 1 if M > 1 else 1
    N = sum(finite)
    lnprobs = np.zeros((M, N, K, 2))
    p_single = np.zeros((MJ, N, K))

    for m, model_name in enumerate(model_names):

        predictor_label_name = config["models"][model_name]["predictor_label_name"]

        y = np.array(data[predictor_label_name][finite])

        model_indices, model_results, gp_parameters, gp_predictions \
            = results["models"][model_name]

        theta, theta_var, \
            mu_single, mu_single_var, \
            sigma_single, sigma_single_var, \
            mu_multiple, mu_multiple_var, \
            sigma_multiple, sigma_multiple_var = gp_predictions.T

        # Calculate probabilities.
        rhos = np.corrcoef(model_results.T)

        print(f"Calculating probabilities for {model_name}")

        for n in tqdm.tqdm(range(N)):
        
            #TODO SKIPPING
            if not np.all(np.isfinite(gp_predictions[n])): continue

            
            mu = gp_predictions[n, ::2]
            diag = np.atleast_2d(gp_predictions[n, 1::2])**0.5

            cov = diag * rhos * diag.T

            draws = np.random.multivariate_normal(mu, cov, size=K).T

            # Clip things to bounded values.
            draws[0] = np.clip(draws[0], 0.5, 1.0)
            draws[1] = np.clip(draws[1], 0.5, 15)
            draws[2] = np.clip(draws[2], 0.05, 10)
            draws[4] = np.clip(draws[4], 0.2, 1.6)

            draws_3_min = np.log(draws[1] + 5 * draws[2]) + draws[4]**2
            draws[3] = np.max([draws[3], draws_3_min], axis=0)

            lnprobs[m, n, :, 0] = np.log(draws[0]) + npm.normal_lpdf(y[n], draws[1], draws[2])
            lnprobs[m, n, :, 1] = np.log(1 - draws[0]) + npm.lognormal_lpdf(y[n], draws[3], draws[4])

            p_single[m, n] = np.exp(lnprobs[m, n, :, 0] - logsumexp(lnprobs[m, n], axis=1))

            """
            
            xi = np.linspace(0, 20, 1000)

            y_s = npm.norm_pdf(xi, np.mean(draws[1]), np.mean(draws[2]), np.mean(draws[0]))
            y_m = npm.lognorm_pdf(xi, np.mean(draws[3]), np.mean(draws[4]), np.mean(draws[0]))

            p_single = np.exp(np.log(y_s) - logsumexp([np.log(y_s), np.log(y_m)], axis=0))

            fig, ax = plt.subplots()
            ax.plot(xi, y_s, c="tab:blue")
            ax.plot(xi, y_m, c="tab:red")
            ax.plot(xi, p_single, c="#000000")

            ax.set_title(n)
            ax.axvline(y[n], c="#666666")

            """


    # Calculate joint probabilities.
    if M > 1:
        print("Calculating joint probabilities")
    
        for n in tqdm.tqdm(range(N)):

            # TODO: SkIPPING
            if not np.all(np.isfinite(gp_predictions[n])): continue
                
            for k in range(K):
                # TODO: this could be wrong,..
                numerator = logsumexp(lnprobs[:, n, k, 0])
                denominator = logsumexp(lnprobs[:, n, k, 1])

                p_single[-1, n, k] = np.exp(numerator - logsumexp([numerator, denominator]))

            """
            numerator = np.sum(lnprobs[:, d, :, 0], axis=0)
            denominator = np.sum(lnprobs[:, d, :, 1], axis=0)
            p_single[-1, d, :] = np.exp(numerator - logsumexp([numerator, denominator], axis=1))
            """


    print("Calculating percentiles")
    percentiles = [5, 50, 95]
    P = len(percentiles)
    p_single_percentiles = np.zeros((P, MJ, N))
    for m in tqdm.tqdm(range(MJ)):
        for n in range(N):

            if not np.all(np.isfinite(gp_predictions[n])): continue
            
            p_single_percentiles[:, m, n] = np.percentile(p_single[m, n], percentiles)


    # Do a classification for each star.
    print("Classifying")
    confidence = np.sum(p_single > 0.5, axis=2)/K
    is_single = confidence > 0.5
    confidence[~is_single] = 1 - confidence[~is_single]

    print("Aggregating data")

    properties = OrderedDict()
    properties["source_id"] = data["source_id"][finite]
    for label_name in all_label_names:
        properties[label_name] = data[label_name][finite]

    # Do GP predictions.
    for m, model_name in enumerate(model_names):

        print(f"Aggregating predictions for {model_name}")
    
        _, __, ___, gp_predictions = results["models"][model_name]
        theta, theta_var, \
            mu_single, mu_single_var, \
            sigma_single, sigma_single_var, \
            mu_multiple, mu_multiple_var, \
            sigma_multiple, sigma_multiple_var = gp_predictions.T

        predictor_label_name = config["models"][model_name]["predictor_label_name"]
        
        properties[predictor_label_name] = np.array(data[predictor_label_name])[finite]

        properties[f"{model_name}_gp_theta"] = theta
        properties[f"{model_name}_gp_mu_s"] = mu_single
        properties[f"{model_name}_gp_sigma_s"] = sigma_single
        properties[f"{model_name}_gp_mu_m"] = mu_multiple
        properties[f"{model_name}_gp_sigma_m"] = sigma_multiple

        #properties[f"{model_name}_gp_theta_var"] = theta_var
        #properties[f"{model_name}_gp_mu_s_var"] = mu_single_var
        #properties[f"{model_name}_gp_sigma_s_var"] = sigma_single_var
        #properties[f"{model_name}_gp_mu_m_var"] = mu_multiple_var
        #properties[f"{model_name}_gp_sigma_m_var"] = sigma_multiple_var

        properties[f"{model_name}_is_single"] = is_single[m].astype(int)
        properties[f"{model_name}_confidence"] = confidence[m]

        for p, percentile in enumerate(percentiles):
            properties[f"{model_name}_p{percentile:.0f}"] = p_single_percentiles[p, m]

    # Joint probabilities.
    if M > 1:
        properties["joint_is_single"] = is_single[-1].astype(int)
        properties["joint_confidence"] = confidence[-1]

        for p, percentile in enumerate(percentiles):
            properties[f"joint_p{percentile:.0f}"] = p_single_percentiles[p, -1]

    print("Writing output file..")
    Table(data=properties).write(output_path, overwrite=True)

    del properties
    print(f"Output written to {output_path}")
