"""
Setup and run the non-parametric model as a map-reduce job.
"""

import numpy as np
import os
import logging
import pickle
import yaml
from astropy.io import fits

import npm_utils as npm

with open(npm.CONFIG_PATH, "r") as fp:
    config = yaml.load(fp)

if not os.path.exists(config["results_path"]):
    os.mkdir(config["results_path"])

# Load in the data.
data = fits.open(config["data_path"])[1].data

# Set up a KD-tree.
X = np.vstack([data[ln] for ln in config["kdtree_label_names"]]).T
finite = np.where(np.all(np.isfinite(X), axis=1))[0]

X = X[finite]
N, D = X.shape

kdt, scale, offset = npm.build_kdtree(
    X, relative_scales=config.get("kdtree_relative_scales", None))

query_kwds = dict(offset=offset, scale=scale, full_output=False)
query_kwds.update(
    minimum_points=config.get("kdtree_minimum_points", 1024), # DEFAULT
    minimum_radius=config.get("kdtree_minimum_radius", None), # DEFAULT
)

# For each point, get the indices needed for that star, ensuring that the KD-tree
# ball follows certain properties (e.g., has enough stars).
for i in range(N):

    # Get indices relative to the finite X
    finite_indices = npm.query_around_point(kdt, X[i], **query_kwds)

    # Translate those finite indices to the original data file, since that is
    # what will be used by the map-jobs to run the optimisation.
    indices = finite[finite_indices]

    # Save the indices to disk.
    path = npm.get_indices_path(data["source_id"][i], config)
    with open(path, "wb") as fp:
        pickle.dump(indices, fp, -1)

    logging.info("{}/{}: Saved {} nearby points to {}".format(
        i, N, indices.size, path))


