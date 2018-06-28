"""
Setup and run the non-parametric model as a map-reduce job.
"""

import numpy as np
import os
import logging
import multiprocessing as mp
import pickle
import yaml
from astropy.io import fits
from time import time

import npm_utils as npm

with open(npm.CONFIG_PATH, "r") as fp:
    config = yaml.load(fp)

if not os.path.exists(config["results_path"]):
    os.mkdir(config["results_path"])

# Load in the data.
data = fits.open(config["data_path"])[1].data

all_label_names = list(config["kdtree_label_names"]) \
                + list(config["predictor_label_names"])

# Set up a KD-tree.
X = np.vstack([data[ln] for ln in config["kdtree_label_names"]]).T
finite = np.where(np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0))[0]

X = X[finite]
N, D = X.shape

kdt, scale, offset = npm.build_kdtree(
    X, relative_scales=config.get("kdtree_relative_scales", None))

kwds = dict(offset=offset, scale=scale, full_output=False)
kwds.update(
    minimum_radius=config.get("kdtree_minimum_radius", None), # DEFAULT
    minimum_points=config.get("kdtree_minimum_points", 1024), # DEFAULT
    maximum_points=config.get("kdtree_maximum_points", 8192), # DEFAULT
)
overwrite = config.get("overwrite_indices", False)

def setup_point(index):

    path = npm.get_indices_path(data["source_id"][index], config)
    
    if os.path.exists(path) and not overwrite: 
        logging.info("Skipping {}/{} because {} exists".format(index, N, path))
        return None

    # Get indices relative to the finite X
    finite_indices = npm.query_around_point(kdt, X[index], **kwds)

    # Translate those finite indices to the original data file, since that is
    # what will be used by the map-jobs to run the optimisation.
    indices = finite[finite_indices]

    # Save the indices to disk.
    with open(path, "wb") as fp:
        pickle.dump(indices, fp, -1)

    logging.info("{}/{}: Saved {} nearby points to {}".format(
        index, N, indices.size, path))

    return None


t_init = time()

processes = mp.cpu_count()
with mp.Pool(processes) as pool:
    pool.map(setup_point, range(N))

logging.info("Setup completed in {:.0f}s".format(time() - t_init))
