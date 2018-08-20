import pickle
import yaml
import numpy as np
from astropy.io import fits
from astropy.table import Table


with open("data/the-battery-stars.indices.pkl", "rb") as fp:
    battery_star_indices = pickle.load(fp)


with open("results/the-battery-stars.astrometric_unit_weight_error.pkl", "rb") as fp:
    astrometric_results = pickle.load(fp)

with open("results/the-battery-stars.rv_single_epoch_scatter.pkl", "rb") as fp:
    rv_results = pickle.load(fp)


with open("the-battery-stars.astrometry.yaml", "r") as fp:
    config = yaml.load(fp)

# Load in the data.
data = fits.open(config["data_path"])[1].data


subset = Table(data[battery_star_indices])

keys = ("theta", "mu_single", "sigma_single", "mu_multi", "sigma_multi")

for i, key in enumerate(keys):
    subset["rv_{}".format(key)] = rv_results[battery_star_indices, i]
    subset["astrometric_{}".format(key)] = astrometric_results[battery_star_indices, i]

finite = np.isfinite(subset["astrometric_theta"] * subset["rv_theta"])
subset = subset[finite]
print(len(subset))

subset.write("results/the-battery-stars.results.fits", overwrite=True)
