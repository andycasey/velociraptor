
"""
Run this after run_analysis.py
"""

import numpy as np
import sys
import pickle
from scipy.special import logsumexp
from astropy.io import fits


def p_single(y, theta, s_mu, s_sigma, b_mu, b_sigma):
    
    s_ivar = s_sigma**-2
    b_ivar = b_sigma**-2
    hl2p = 0.5 * np.log(2*np.pi)
    
    s_lpdf = -hl2p + 0.5 * np.log(s_ivar) \
           - 0.5 * (y - s_mu)**2 * s_ivar
    
    b_lpdf = -np.log(y*b_sigma) - hl2p \
           - 0.5 * (np.log(y) - b_mu)**2 * b_ivar

    foo = np.vstack([s_lpdf, b_lpdf]).T + np.log([theta, 1-theta])
    p_single = np.exp(foo.T[0] - logsumexp(foo, axis=1))

    return p_single




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

    for results in all_results:

        config = results["config"]
        data = fits.open(config["data_path"])[1].data

        all_label_names = list(config["kdtree_label_names"]) \
                        + list(config["require_finite_label_names"]) \
                        + list(config["predictor_label_names"])
        all_label_names = list(np.unique(all_label_names))     

        # Set up a KD-tree.
        finite = np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0)
        y = np.array([data[ln][finite] for ln in config["predictor_label_names"]]).flatten()



        # Calculate the mu_multiple.
        mu_single, mu_single_var, sigma_single, sigma_single_var, \
            sigma_multiple, sigma_multiple_var = results["gp_predictions"].T

        mu_multiple = np.log(mu_single + sigma_single) \
                    + sigma_multiple**2

        # Calculate probabilities.
        mean_theta = np.mean(results["results"][:, 0])

        foo = p_single(y, mean_theta, mu_single, sigma_single, mu_multiple, sigma_multiple)

        # Do draws of say 10 objects.
        D = 10
        mu_single, mu_single_var = (mu_single[:D], mu_single_var[:D])
        sigma_single, sigma_single_var = (sigma_single[:D], sigma_single_var[:D])
        sigma_multiple, sigma_multiple_var = (sigma_multiple[:D], sigma_multiple_var[:D])

        K = 1000
        foos = np.zeros((K, D))
        for k in range(K):
            ms = np.random.normal(mu_single, mu_single_var**0.5)
            ss = np.random.normal(sigma_single, sigma_single_var**0.5)
            sm = np.random.normal(sigma_multiple, sigma_multiple_var**0.5)
            mm = np.log(ms + ss) + sm**2

            foos[k, :] = p_single(y[:D], mean_theta, ms, ss, mm, sm)


        percents = np.nanpercentile(foos, [16, 50, 84], axis=0)

        raise a




# Need:
# result_paths

# Check seed for each one and ensure they are the same.

# Calculate the mu_multiple (as the lower value).

# Calculate probabilities for each source, after propagating variance.