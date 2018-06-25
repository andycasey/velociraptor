"""
Setup and run the non-parametric model as a map-reduce job.
"""

import numpy as np
import os
import logging
import pickle
from astropy.table import Table

import npm_utils

DATA_PATH = "data/rv-all-subset-1e4.fits"
RESULTS_PATH = "results/"

# indices only: 153 Mb for 10^4 sources.
# indices only: 153 Gb for 10^7 sources.

kdtree_label_names = ("bp_rp", "absolute_rp_mag", "phot_rp_mean_mag")

get_indices_path = lambda source_id: os.path.join(
    RESULTS_PATH, "{}.input".format(source_id))

if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)


# Load in the data.
data = Table.read(DATA_PATH)

# Set up a KD-tree.
X = np.vstack([data[ln] for ln in kdtree_label_names]).T
finite = np.where(np.all(np.isfinite(X), axis=1))[0]

X = X[finite]
N, D = X.shape

kdtree, scale, offset = npm_utils.build_kdtree(X)

query_kwds = dict(offset=offset, scale=scale, minimum_points=1000,
                  minimum_radius=[0.1, 1, 1], full_output=False)

# For each point, get the indices needed for that star, ensuring that the KD-tree
# ball follows certain properties (e.g., has enough stars).
for i in range(N):

    # Get indices relative to the finite X
    finite_indices = npm_utils.query_around_point(kdtree, X[i], **query_kwds)

    # Translate those finite indices to the original data file, since that is
    # what will be used by the map-jobs to run the optimisation.
    indices = finite[finite_indices]

    # Put the point of interest as the first index.
    index = np.where(finite[i] == indices)[0][0]
    indices[index] = indices[0]
    indices[0] = finite[i]

    # Save the indices to disk.
    path = get_indices_path(data["source_id"][i])
    with open(path, "wb") as fp:
        pickle.dump(indices, fp, -1)

    logging.info("{}/{}: Saved {} nearby points to {}".format(
        i, N, indices.size, path))


