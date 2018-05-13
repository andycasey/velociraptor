
"""
Fitting and sampling of the radial velocity variance model.
"""

import pickle
import velociraptor

# Load the data and make some plots.
sources = velociraptor.load_gaia_sources("data/rv-all.fits")
model, data_dict, init_dict, idx = velociraptor.prepare_model(N=1e6, **sources)

print("Number of data points: {}".format(data_dict["N"]))

p_opt = model.optimizing(data=data_dict, init=init_dict)

with open("model-optimized.pkl", "wb") as fp:
    pickle.dump(p_opt, fp, -1)

print(p_opt)

sampled_model = model.sampling(**velociraptor.stan.sampling_kwds(
    data=data_dict, chains=2, iter=2000, init=p_opt))

samples = sampled_model.extract(("theta", "mu_coefficients", 
    "sigma_coefficients", "log_membership_probability"))

with open("model-sampled.pkl", "wb") as fp:
    pickle.dump(samples, fp, -1)