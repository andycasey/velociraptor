
"""
Run this after run_analysis.py
"""

import logging
import numpy as np
import sys
import pickle
import tqdm
from scipy.special import logsumexp
from astropy.io import fits

def ps(y, theta, s_mu, s_sigma, b_mu, b_sigma):
    
    s_ivar = s_sigma**-2
    b_ivar = b_sigma**-2
    hl2p = 0.5 * np.log(2*np.pi)
    
    s_lpdf = -hl2p + 0.5 * np.log(s_ivar) \
           - 0.5 * (y - s_mu)**2 * s_ivar
    
    b_lpdf = -np.log(y*b_sigma) - hl2p \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar
    b_lpdf[~np.isfinite(b_lpdf)] = -np.inf

    foo = np.vstack([s_lpdf, b_lpdf]).T + np.log([theta, 1-theta])
    p_single = np.exp(foo.T[0] - logsumexp(foo, axis=1))
    assert np.all(np.isfinite(p_single))
    return p_single

def lnprob_single(y, theta, s_mu, s_sigma, b_mu, b_sigma):

    s_ivar = s_sigma**-2
    b_ivar = b_sigma**-2
    hl2p = 0.5 * np.log(2*np.pi)
    
    s_lpdf = -hl2p + 0.5 * np.log(s_ivar) \
           - 0.5 * (y - s_mu)**2 * s_ivar
    
    b_lpdf = -np.log(y*b_sigma) - hl2p \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar
    b_lpdf[~np.isfinite(b_lpdf)] = -np.inf

    return np.vstack([s_lpdf, b_lpdf]).T + np.log([theta, 1-theta])



if __name__ == "__main__":

    result_paths = sys.argv[1:]

    if len(result_paths) < 1:
        raise ValueError("no result paths given")

    all_results = []
    for result_path in result_paths:
        logging.info(f"Loading results from {result_path}")

        with open(result_path, "rb") as fp:
            r = pickle.load(fp)

        all_results.append(r)

    # Check seeds are the same.
    initial_seed = all_results[0]["seed"]
    if initial_seed is None:
        raise ValueError("seed is not known")

    for r in all_results:
        if int(r["seed"]) != int(initial_seed):
            raise ValueError(f"seeds differ: {initial_seed} in {result_paths[0]} != {r['seed']}")

    # TODO: check that the data files etc are the same between results paths
    # TODO: this is so hacky just re-factor the model format and run_analysis.py

    results = all_results[0]

    config = results["config"]
    data = fits.open(config["data_path"])[1].data

    all_label_names = list(config["kdtree_label_names"]) \
                    + list(config["require_finite_label_names"]) \
                    + list(config["predictor_label_names"])
    all_label_names = list(np.unique(all_label_names))     

    # Set up a KD-tree.
    finite = np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0)

    y = np.array([data[ln][finite] for ln in config["predictor_label_names"]]).flatten()

    K = 100
    M = len(result_paths)
    lnprobs = np.zeros((M, y.size, K, 2))
    p_single = np.zeros((M + 1, y.size, K))

    for m, (result_path, results) in enumerate(zip(result_paths, all_results)):

        mu_single, mu_single_var, sigma_single, sigma_single_var, \
            sigma_multiple, sigma_multiple_var = results["gp_predictions"].T

        # Calculate probabilities.
        # TODO MAGIC HACK ALL DIRECTLY BELOW
        mean_theta = np.mean(results["results"][:, 0])
        
        mask = np.array([1, 2, 4])
        rhos = np.corrcoef(results["results"][:, mask].T)

        logging.info(f"Calculating probabilities for {result_path}")

        for d in tqdm.tqdm(range(y.size)):
        
            diag = np.atleast_2d(np.array([
                mu_single_var[d], 
                sigma_single_var[d], 
                sigma_multiple_var[d]
            ]))**0.5

            cov = diag * rhos * diag.T

            mu = np.array([mu_single[d], sigma_single[d], sigma_multiple[d]])

            ms, ss, sm = np.random.multivariate_normal(mu, cov, size=K).T
            # Calculate the mu_multiple. TODO HACK MAGIC

            mm = np.log(ms + ss) + sm**2

            lnprobs[m, d] = lnprob_single(y[[d]], mean_theta, ms, ss, mm, sm)
            p_single[m, d] = np.exp(lnprobs[m, d, :, 0] - logsumexp(lnprobs[m, d], axis=1))

    # Calculate joint probabilities.
    for d in tqdm.tqdm(range(y.size)):
        for k in range(K):
            foo = lnprobs[:, d, k]

            numerator = logsumexp(lnprobs[:, d, k, 0])
            denominator = logsumexp(lnprobs[:, d, k, 1])

            p_single[-1, d, k] = np.exp(numerator - logsumexp([numerator, denominator]))

    # Save percentiles of the probability distributions.
    # Save the draws from the probability distributions.
    
    raise a


# Need:
# result_paths

# Check seed for each one and ensure they are the same.

# Calculate the mu_multiple (as the lower value).

# Calculate probabilities for each source, after propagating variance.