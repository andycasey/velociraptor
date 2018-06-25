
import pickle
import logging
import numpy as np
import os
from glob import glob
from time import time
from astropy.io import fits
from random import shuffle

import npm_utils
import stan_utils as stan


# -------------
DATA_PATH = "data/rv-all-subset-1e4.fits"
RESULTS_PATH = "results/"


kwds = dict(predictor_label_names=(
                "rv_single_epoch_scatter",
                "astrometric_unit_weight_error",
                "phot_bp_rp_excess_factor",
                "rv_abs_diff_template_teff",
            ),
            model_path="npm.stan", remove_input_file=True)


def optimize_npm(input_path, predictor_label_names, data, model_path="npm.stan", 
    remove_input_file=False, **kwargs):
    """
    Optimize the two-component model for a source, using the properties of the
    nearby sources.
    """

    with open(input_path, "rb") as fp:
        indices = pickle.load(fp)

    y = np.array([data[ln][indices] for ln in predictor_label_names]).T
    N, D = y.shape

    init_values = npm_utils.get_initialization_point(y)

    init_dict = dict(zip(
        ("theta", "mu_single", "sigma_single", "mu_multiple", "sigma_multiple"),
        npm_utils._unpack_params(init_values)))
    init_dict["mu_multiple_uv"] = 0.5 * np.ones(D)

    data_dict = dict(y=y, N=N, D=D)

    opt_kwds = dict(
        data=data_dict,
        init=init_dict,
        verbose=False,
        tol_obj=7./3 - 4./3 - 1, # machine precision
        tol_grad=7./3 - 4./3 - 1, # machine precision
        tol_rel_grad=1e3,
        tol_rel_obj=1e4,
        iter=10000)
    opt_kwds.update(kwargs)

    model = stan.load_stan_model(model_path)
    
    t_init = time()
    p_opt = model.optimizing(**opt_kwds)
    t_complete = time() - t_init

    # Calculate membership probability.
    p_single = npm_utils.membership_probability(y[0], p_opt)

    # Calculate excess RV scatter.
    li = list(predictor_label_names).index("rv_single_epoch_scatter")
    rv_excess, rv_excess_significance = npm_utils.label_excess(y[0], p_opt, li)

    result = dict(
        p_opt=p_opt, p_single=p_single, 
        rv_excess=rv_excess, rv_excess_significance=rv_excess_significance,
        points_in_ball=N)
    
    logging.info("Result from {}: {}".format(input_path, result))

    output_path = input_path.replace(".input", ".output")
    with open(output_path, "wb") as fp:
        pickle.dump(result, fp, -1)

    logging.info("Write output from {} to {} (took {:.0f}s to optimize)".format(
        input_path, output_path, t_complete))

    if remove_input_file:
        logging.info("Removing input file: {}".format(input_path))
        os.unlink(input_path)

    return result



if __name__ == "__main__":

    # Load the data once here.
    image = fits.open(DATA_PATH)
    kwds.update(data=image[1].data)

    input_paths = glob(os.path.join(RESULTS_PATH, "*.input"))
    N = len(input_paths)

    shuffle(input_paths)

    for i, input_path in enumerate(input_paths):
        if not os.path.exists(input_path):
            logging.info("{}/{}: Skipping because {} no longer exists".format(
                i, N, input_path))
            continue

        logging.info(
            "{}/{}: Optimizing from input file: {}".format(i, N, input_path))
        result = optimize_npm(input_path, **kwds)

