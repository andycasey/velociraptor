
"""
Fitting and sampling of the radial velocity variance model.
"""

import numpy as np
import pickle
import velociraptor
from scipy.special import logsumexp

# Load the data and make some plots.
sources = velociraptor.load_gaia_sources("data/rv-all.fits")
model, data_dict, init_dict, idx = velociraptor.prepare_model(S=1e5, **sources)

print("Number of data points: {}".format(data_dict["N"]))

init_dict = dict([
    ('theta', 0.15167079461165178),
    ('mu_coefficients', np.array([2.1465e-05, 1.4585e+02, 2.0827e+11,
        7.7332e-08, 5.8626e+00])), 
    ('sigma_coefficients', np.array([7.4405e-04, 3.9762e-01, 1.7293e+11,
        4.1103e-04, 5.9489e+00])),
])

p_opt = model.optimizing(data=data_dict, init=init_dict)

with open("model-optimized.pkl", "wb") as fp:
    pickle.dump(p_opt, fp, -1)

print(p_opt)
"""
sampled_model = model.sampling(**velociraptor.stan.sampling_kwds(
    data=data_dict, chains=2, iter=2000, init=p_opt))

samples = sampled_model.extract(("theta", "mu_coefficients", 
    "sigma_coefficients", "log_membership_probability"))

with open("model-sampled.pkl", "wb") as fp:
    pickle.dump(samples, fp, -1)
"""


# Calculate probabilities.
# SB1: From what Gaia doesn't tell us.
is_sb1 = (sources["phot_rp_mean_mag"] <= 12.8) \
       * (~np.isfinite(sources["radial_velocity"]))

sources["p_sb1"] = np.zeros(len(sources), dtype=float)
sources["p_sb1"][is_sb1] = 1.0

# SB2: From our model.
# In case I want to change this later to take the mean from a trace or something
_point_estimate = lambda k: p_opt[k]

log_ps1 = _point_estimate("log_ps1")

# Get the design matrix and single epoch rv variance for ALL stars.
dm = velociraptor._rvf_design_matrix(**sources).T
rv_variance = sources["rv_single_epoch_variance"]

mu = np.dot(dm, _point_estimate("mu_coefficients"))
ivar = np.dot(dm, _point_estimate("sigma_coefficients"))**-2
log_ps2 = np.log(1 - _point_estimate("theta")) \
        - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(ivar) \
        - 0.5 * (rv_variance - mu)**2 * ivar

log_sb2 = log_ps1 - logsumexp([log_ps1 * np.ones(dm.shape[0]), log_ps2], axis=0)
sources["p_sb2"] = np.exp(log_sb2)

# Calculate the max of those two probabilities.
sources["p_sbx"] = np.nanmax([sources["p_sb1"], sources["p_sb2"]], axis=0)

# Calculate the excess variance.
sources["excess_rv_variance"] = np.max(
    [rv_variance - mu, np.zeros(rv_variance.size)], axis=0)
#sources["excess_rv_variance"][~np.isfinite(sources["excess_rv_variance"])] = 0
sources["excess_rv_sigma"] = sources["excess_rv_variance"]**0.5

