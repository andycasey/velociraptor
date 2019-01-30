
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
    N = sum(finite)
    lnprobs = np.zeros((M, N, K, 2))
    p_single = np.zeros((M + 1, N, K))

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
        # TODO MAGIC HACK ALL DIRECTLY BELOW
        
        mask = np.array([1, 2, 4])
        rhos = np.corrcoef(model_results[:, mask].T)

        print(f"Calculating probabilities for {model_name}")

        count = 0
        for n in tqdm.tqdm(range(N)):
        
            #TODO SKIPPING
            if not np.all(np.isfinite(gp_predictions[n])): continue

            diag = np.atleast_2d(np.array([
                mu_single_var[n], 
                sigma_single_var[n], 
                sigma_multiple_var[n]
            ]))**0.5

            cov = diag * rhos * diag.T

            mu = np.array([mu_single[n], sigma_single[n], sigma_multiple[n]])

            ms, ss, sm = np.random.multivariate_normal(mu, cov, size=K).T
            # Calculate the mu_multiple. TODO HACK MAGIC
            ss = np.clip(ss, 1e-3, np.inf)
            sm = np.clip(sm, 1e-3, np.inf)

            mm = np.log(ms + 5 * ss) + sm**2
            #mm = mu_multiple[n]


            #lnprobs[m, n] = lnprob(y[[n]], mean_theta, ms, ss, mm, sm)
            #p_single[m, n] = np.exp(lnprobs[m, n, :, 0] - logsumexp(lnprobs[m, n], axis=1))

            xi = np.linspace(0, 10, 1000)

            
            mean_theta =
            y_s = mean_theta * stats.norm.pdf(xi, np.mean(ms), np.mean(ss))
            y_m_alt = (1 - mean_theta) * stats.lognorm.pdf(np.exp(np.log(xi) - np.mean(mm)), np.mean(sm))

            y_m = (1 - mean_theta) * np.exp(-0.5 * ((np.log(xi) - np.mean(mm))/np.mean(sm))**2) \
                    / (xi * np.mean(sm) * np.sqrt(2 * np.pi))



            y_s2 = np.log(mean_theta) + stats.norm.logpdf(xi, np.mean(ms), np.mean(ss))
            y_m2 = np.log(1 - mean_theta) + stats.lognorm.logpdf(xi, np.mean(mm), np.mean(sm))
            y_m2 = np.log(1 - mean_theta) - 0.5 * ((np.log(xi) - np.mean(mm))/np.mean(sm))**2 \
                    - np.log(xi * np.mean(sm) * np.sqrt(2 * np.pi))

            p_single = np.exp(y_s2 - logsumexp([y_s2, y_m2], axis=0))

            fig, ax = plt.subplots()
            ax.plot(xi, y_s, c="tab:blue")
            ax.plot(xi, y_m, c="tab:red")
            ax.plot(xi, y_m_alt, c="tab:green")
            ax.plot(xi, p_single, c="#000000")

            ax.set_title(n)
            ax.axvline(y[n], c="#666666")

            assert np.all((ms + 2 * ss) >= np.exp(mm - sm**2))

            raise a
            count += 1

            if count > 10:
                raise a

            raise a


    s_ivar = np.abs(s_sigma)**-2
    b_ivar = np.abs(b_sigma)**-2
    hl2p = 0.5 * np.log(2*np.pi)
    
    s_lpdf = -hl2p + 0.5 * np.log(s_ivar) \
           - 0.5 * (y - s_mu)**2 * s_ivar
    
    b_lpdf = -np.log(y*b_sigma) - hl2p \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar
    b_lpdf[~np.isfinite(b_lpdf)] = -np.inf


    # Calculate joint probabilities.
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
    p_single_percentiles = np.zeros((P, M + 1, N))
    for m in tqdm.tqdm(range(M + 1)):
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
    print("Aggregating predictions")
    for m, model_name in enumerate(model_names):

        _, __, gp_predictions = results["models"][model_name]
        mu_single, mu_single_var, sigma_single, sigma_single_var, \
            sigma_multiple, sigma_multiple_var = gp_predictions.T

        predictor_label_name = config["models"][model_name]["predictor_label_name"]
        y = np.array(data[predictor_label_name])[finite]

        properties[predictor_label_name] = y

        properties[f"{model_name}_gp_mu_s"] = mu_single
        properties[f"{model_name}_gp_sigma_s"] = sigma_single
        #properties[f"{model_name}_gp_sigma_m"] = sigma_multiple

        #properties[f"{model_name}_gp_mu_s_var"] = mu_single_var
        #properties[f"{model_name}_gp_sigma_s_var"] = sigma_single_var
        #properties[f"{model_name}_gp_sigma_m_var"] = sigma_multiple_var

        properties[f"{model_name}_is_single"] = is_single[m].astype(int)
        properties[f"{model_name}_confidence"] = confidence[m]

        for p, percentile in enumerate(percentiles):
            properties[f"{model_name}_p{percentile:.0f}"] = p_single_percentiles[p, m]

    # Joint probabilities.
    properties["joint_is_single"] = is_single[-1].astype(int)
    properties["joint_confidence"] = confidence[-1]

    for p, percentile in enumerate(percentiles):
        properties[f"joint_p{percentile:.0f}"] = p_single_percentiles[p, -1]

    Table(data=properties).write(output_path, overwrite=True)

    del properties
    print(f"Output written to {output_path}")
