"""
Script to aggregate results together from the non-parametric model.
"""

import numpy as np
import os
import logging
import pickle
import yaml
from astropy.io import fits
from glob import glob

import npm_utils as npm

with open(npm.CONFIG_PATH, "r") as fp:
    config = yaml.load(fp)

# Load the data frame so that we can get the right index.
image = fits.open(config["data_path"])
data = image[1].data

N = len(data)
    
# Figure out what columns we are going to add to the data.
common_suffix = ""
common_label_names = ["theta"]

output_label_names = ["{0}{1}".format(cln, common_suffix) for cln in common_label_names]

for kind in ("single", "multiple"):
    for prefix in ("mu", "sigma"):
        for predictor_label_name in config["predictor_label_names"]:
            output_label_names.append("_".join([prefix, kind, predictor_label_name]))


K = len(output_label_names)
L = len(config["predictor_label_names"])

outputs = np.nan * np.ones((N, K), dtype=float)

for index in range(N):

    output_path = npm.get_output_path(data["source_id"][index], config,
                                      check_path_exists=False)
    
    logging.info("({}/{}): Aggregating results from {}".format(
        index, N, output_path))

    if os.path.lexists(output_path) and not os.path.exists(output_path):
        raise a

    if not os.path.exists(os.path.realpath(os.path.abspath(output_path))):
        realpath = os.path.realpath(output_path)
        logging.info("Skipping because {} does not exist ({} -> {})".format(
            output_path, output_path, realpath))
        continue

    with open(output_path, "rb") as fp:
        result = pickle.load(fp)

    # Parse the output from the result file.
    p_opt = result["p_opt"]
    theta = p_opt["theta"]

    offset = 1
    outputs[index, 0] = theta
    
    for k, key in enumerate(("mu_single", "sigma_single", "mu_multiple", "sigma_multiple")):
        outputs[index, offset + k * L:offset + (k + 1) * L] = p_opt[key]
    

# Add these columns to the existing data file?
columns = fits.ColDefs(
    [c for c in image[1].data.columns] + \
    [fits.Column(name=pln, format="E", array=outputs[:, i]) for i, pln in enumerate(output_label_names)]
)
# Warning: this is going to create an in-memory copy of the data!
#          we may need to revisit this if it kills us.
image[1] = fits.BinTableHDU.from_columns(columns)
image.writeto(config["output_data_path"], overwrite=True)

logging.info("Wrote output file to {}".format(config["output_data_path"]))