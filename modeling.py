
"""
Modeling and validation of the radial velocity calibration model.
"""

import numpy as np
import matplotlib.pyplot as plt

import velociraptor

# Load the data and make some plots.
sources = velociraptor.load_gaia_sources("data/rv-cal-subset.fits.gz", N=1000)

model, data_dict, init_dict, used_in_fit = velociraptor.prepare_model(
    sources["phot_rp_mean_flux"], sources["rv_single_epoch_variance"])

samples = model.sampling(**velociraptor.stan.sampling_kwds(
    data=data_dict, chains=2, iter=2000, init=init_dict))





scatter_kwds = dict(s=1, facecolor="#000000", alpha=0.5, rasterized=True)


fig, ax = plt.subplots()

indices = np.argsort(sources["phot_rp_mean_flux"])
x = sources["phot_rp_mean_flux"][indices]
mu, sigma = velociraptor.predict_map_rv_single_epoch_variance(samples, x)

ax.scatter(
    sources["phot_rp_mean_flux"], sources["rv_single_epoch_variance"],
    **scatter_kwds)
ax.plot(x, mu, "r-")
ax.fill_between(x, mu - sigma, mu + sigma, 
    facecolor="r", alpha=0.3, edgecolor="none")
ax.set_ylim(-0.5, 10)
ax.semilogx()



fig, ax = plt.subplots()

ax.scatter(
    x, sources["rv_single_epoch_variance"] - expectation,
    **scatter_kwds)

ax.semilogx()



