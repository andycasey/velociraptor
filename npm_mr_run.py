
import pickle
import logging
import numpy as np
import os
import yaml
from glob import glob
from time import time
from astropy.io import fits
from random import shuffle

import npm_utils as npm
import stan_utils as stan



def initialize_from_nearby_point(indices, data, config):
    """
    Look for an initialization point from nearby results.
    """
    raise NotImplementedError
    return None

    for index in indices:
        source_id = data["source_id"][index]
        output_path = npm.get_output_path(source_id, config)
        if os.path.exists(output_path):

            with open(output_path, "rb") as fp:
                result = pickle.load(fp)

            # Ensure that we have the right dimensions.
            p_opt = result["p_opt"]
            for k in ("mu_single", "sigma_single", "mu_multiple", 
                      "sigma_multiple", "mu_multiple_uv"):
                p_opt[k] = np.atleast_1d(p_opt[k])

            return (source_id, p_opt)

    # Nothing!
    return None




def optimize_npm_at_point(index, indices, data, config):

    # Get indices and load data.
    y = np.array([data[ln][indices] for ln in config["predictor_label_names"]]).T
    N, D = y.shape

    init_dict = initialize_from_nearby_point(indices, data, config)

    if init_dict is None:
        logging.info("Doing pre-initialization based on the data.")
        init_from_source_id = -1
        init_dict = npm.get_initialization_point(y)
        
    else:
        init_from_source_id, init_dict = init_dict
        logging.info("Initializing from result obtained for source_id: {}"\
                     .format(init_from_source_id))

    opt_kwds = dict(
        data=dict(y=y, N=N, D=D),
        init=init_dict,
        verbose=False,
        tol_obj=7./3 - 4./3 - 1, # machine precision
        tol_grad=7./3 - 4./3 - 1, # machine precision
        tol_rel_grad=1e3,
        tol_rel_obj=1e4,
        iter=10000)
    opt_kwds.update(config.get("optimisation_kwds", {}))

    # Make sure that some entries have the right units.
    for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
        if key in opt_kwds:
            opt_kwds[key] = float(opt_kwds[key])

    model = stan.load_stan_model(config["model_path"])

    S = config.get("share_optimised_result_with_nearest", 0)
    
    t_init = time()
    try:
        p_opt = model.optimizing(**opt_kwds)
    
    except:
        logging.exception("Failed to optimize from {}".frmat(indices_path))
        return (indices, S, None)

    t_complete = time() - t_init

    result = dict(p_opt=p_opt, 
                  data_index=data_index, 
                  init_from_source_id=init_from_source_id)

    logging.info("p_opt ({0:.0f}s): {1}".format(t_complete, p_opt))
    logging.info("Writing to {}".format(output_path))

    with open(output_path, "wb") as fp:
        pickle.dump(result, fp, -1)

    # Assign this result to nearby stars?
    logging.info("Sharing result with nearest {} neighbours".format(S))


    for nearby_index in indices[:1 + S]:
        if nearby_index == data_index: continue
        
        nearby_output_path = npm.get_output_path(data["source_id"][nearby_index],
                                                 config)
        if os.path.lexists(nearby_output_path):
            os.unlink(nearby_output_path)

        os.symlink(os.path.abspath(output_path), 
                   os.path.abspath(nearby_output_path))

        logging.info("Shared result with neighbour {} -> {}".format(
            output_path, nearby_output_path))


    return (indices, S, p_opt)


def select_next_point(indices, data, config):

    for index in indices:
        path = npm.get_output_path(data["source_id"][index], config)

        # Check for true files or symbolic links.
        if not os.path.lexists(path):
            return index

    return None





if __name__ == "__main__":

    with open(npm.CONFIG_PATH, "r") as fp:
        config = yaml.load(fp)


    # Load the data.
    data = fits.open(config["data_path"])[1].data
    N = len(data)


    indices_kwds = dict(filename=config["indices_path"], dtype=np.int32)
    indices_bpr = config["kdtree_maximum_points"] \
                * np.dtype(indices_kwds["dtype"]).itemsize



    # Generator for a random index to run when the swarm gets bored.
    def indices():
        yield from np.random.choice(N, N, replace=False)

    def run_swarm():

        count = 0
        for jumps, index in enumerate(indices()):

            # Swarm strategy.
            while True:
                
                # Get indices for this index.
                indices = np.memmap(mode="r", 
                                    shape=(config["kdtree_maximum_points"], ),
                                    offset=index * indices_bpr,
                                    **indices_kwds)

                if not any(fp > 0):
                    # No indices. Skip.
                    # TODO: update output file with nans at this row?
                    break
                
                
                S, p_opt = optimize_npm_at_point(index,
                                                 indices,
                                                 data,
                                                 config)

                # Select a new point to run on.
                index = select_next_point(indices[1 + S:], data, config)

                if index is None:
                    logging.info("All nearby swarm points are done! "\
                                 "Selecting new point to start from.")
                    break # out to the generator


