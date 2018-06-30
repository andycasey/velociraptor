"""
Setup and run the non-parametric model as a map-reduce job.
"""

import numpy as np
import os
import multiprocessing as mp
import yaml
from astropy.io import fits
import tqdm

import npm_utils as npm

with open(npm.CONFIG_PATH, "r") as fp:
    config = yaml.load(fp)

# Load in the data.
data = fits.open(config["data_path"])[1].data

all_label_names = list(config["kdtree_label_names"]) \
                + list(config["predictor_label_names"])

# Set up a KD-tree.
X = np.vstack([data[ln] for ln in config["kdtree_label_names"]]).T
finite = np.where(np.all([np.isfinite(data[ln]) for ln in all_label_names], axis=0))[0]
N, D = X.shape
F = finite.size

kdt, scale, offset = npm.build_kdtree(
    X[finite], relative_scales=config.get("kdtree_relative_scales", None))

kwds = dict(offset=offset, scale=scale, full_output=False)
kwds.update(
    minimum_radius=config.get("kdtree_minimum_radius", None), # DEFAULT
    minimum_points=config.get("kdtree_minimum_points", 1024), # DEFAULT
    maximum_points=config.get("kdtree_maximum_points", 8192), # DEFAULT
)

# Prepare the memory mapped array.
memmap_kwds = dict(filename=config["indices_path"], dtype=np.int32)
if not os.path.exists(memmap_kwds["filename"]):
    fp = np.memmap(mode="w+", shape=(N, kwds["maximum_points"]), **memmap_kwds)
    del fp

# Calculate the number of bytes per row.
bpr = kwds["maximum_points"] * np.dtype(memmap_kwds["dtype"]).itemsize


def setup_point(index):
    fp = np.memmap(mode="r+", shape=(kwds["maximum_points"], ),
                   offset=index * bpr, **memmap_kwds)

    if any(fp > 0):
        del fp
        return None

    # Get indices.
    indices = npm.query_around_point(kdt, X[index], **kwds)
    
    # Write the result to the right place in memory and make indices relative to 
    # the original data file
    P = indices.size
    fp[:P] = finite[indices]
    fp[P:] = -1
    fp.flush()
    del fp

    return None

with mp.Pool(processes=200) as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(setup_point, finite), total=F):
        pass
