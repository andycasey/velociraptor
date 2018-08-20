import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import numpy as np
import os
import pickle
import yaml
import scipy.optimize as op
from astropy.io import fits
from matplotlib.ticker import MaxNLocator
from time import time
from scipy.special import logsumexp

import george
import george.kernels

from mpl_utils import mpl_style


seed = 123
np.random.seed(seed)
matplotlib.style.use(mpl_style)

gp_labels = (
    r"$\mu_\textrm{s}$ \textrm{/ km\,s}$^{-1}$",
    r"$\sigma_\textrm{s}$ \textrm{/ km\,s}$^{-1}$",
    r"$\mu_\textrm{m}$ \textrm{/ km\,s}$^{-1}$",
    r"$\sigma_\textrm{m}$ \textrm{/ km\,s}$^{-1}$",
)
G = len(gp_labels)



def x_for_gp(x):
    x_ = np.copy(x)
    x_[:, 0] = 1.0/x_[:,0]
    return x_


def construct_kernel(D):
    return george.kernels.ExpKernel(np.ones(D), ndim=D) \
         + george.kernels.Matern32Kernel(np.ones(D), ndim=D)


# Load the data.
with open("npm-config.rv.yaml", "r") as fp:
    config = yaml.load(fp)

with open("results/rv_single_epoch_scatter.v3.pkl", "rb") as fp:
    npm_results = pickle.load(fp)


data = fits.open(config["data_path"])[1].data

X = np.vstack([data[ln] for ln in config["kdtree_label_names"]]).T

# Number of points to fit the GPs.
M = 1000


N, D = X.shape
finite_indices = np.arange(N)[np.all(np.isfinite(npm_results), axis=1)]

fit_indices = np.random.choice(finite_indices, M, replace=False)


gps = []
x = X[fit_indices]

for npm_index in range(1, 5):

    print("Fitting GP to npm index {}".format(npm_index))

    y = npm_results[fit_indices, npm_index]

    kernel = construct_kernel(D)

    gp = george.GP(kernel, mean=np.mean(y), white_noise=np.log(np.std(y)),
                   fit_mean=True, fit_white_noise=True)

    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    gp.compute(x_for_gp(x))

    print("Initial \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
    print("initial \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

    p0 = gp.get_parameter_vector()

    t_init = time()
    result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    t_opt = time() - t_init

    print("Result: {}".format(result))
    print("Final \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
    print("Took {:.0f} seconds to optimize".format(t_opt))

    gps.append((gp.get_parameter_dict(), y))


with open("gp-rv-hps.pkl", "wb") as fp:
    pickle.dump((x, gps), fp, -1)




with open("gp-rv-hps.pkl", "rb") as fp:
    x, gps = pickle.load(fp)




K = 100000  #how many points to predict for.
predict_indices = np.random.choice(finite_indices, K, replace=False)
x_pred = X[predict_indices]

fig, axes = plt.subplots(2, len(gp_labels))
axes = np.atleast_2d(axes).T



ylabels = (
    r"\textrm{phot rp mean mag}",
    r"\textrm{absolute rp mag}"
)

scatter_kwds = dict(s=1, alpha=0.5)

_, D = x_pred.shape

for i, (ax_row, gp_label, (hp, y)) \
in enumerate(zip(axes, gp_labels, gps)):

    print("Predicting properties for {} (index {})".format(gp_labels[i], i))

    kernel = construct_kernel(D)

    #white_noise=-50,
    gp = george.GP(kernel, mean=0.0, white_noise=0.0,
                   fit_mean=True, fit_white_noise=True)
    gp.set_parameter_vector(list(hp.values()))

    # Compute the Gaussian Process at the given x values.
    gp.compute(x_for_gp(x))

    # Make predictions using the GP, conditioned on the y values.
    pred, pred_var = gp.predict(y, x_for_gp(x_pred), return_var=True)

    print(np.percentile(pred, [0, 16, 50, 84, 100]))

    for j, (ax, ylabel) in enumerate(zip(ax_row, ylabels)):
        ax.scatter(x_pred.T[0], x_pred.T[1 + j], c=pred, **scatter_kwds)

        ax.set_xlabel(r"\textrm{bp - rp}")
        ax.set_ylabel(ylabel)

        ax.set_title(gp_label)


for ax_row in axes.T:
    ylims = np.array([ax.get_ylim() for ax in ax_row])
    ylims = (np.min(ylims), np.max(ylims))

    for ax in ax_row:
        ax.set_ylim(ylims[::-1])


# Do predictions for all the things.
X_finite = np.arange(N)[np.all(np.isfinite(X), axis=1)]
x_pred = X[X_finite]

G = len(gps)
# value, variance for each gaussian process, plus
# log-likelihood for single star model
# log-likelihood for binary star model
# LL ratio.

gpm_predictions = np.nan * np.ones((N, 2 * G + 3))

for i, (hp, y) in enumerate(gps):

    kernel = construct_kernel(D)

    #white_noise=-50,
    gp = george.GP(kernel, mean=0.0, white_noise=0.0,
                   fit_mean=True, fit_white_noise=True)
    gp.set_parameter_vector(list(hp.values()))

    # Compute the Gaussian Process at the given x values.
    gp.compute(x_for_gp(x))

    # Make predictions using the GP, conditioned on the y values.
    # Do it in chunks...
    chunk_size = 100000
    K = int(np.ceil(X_finite.size / float(chunk_size)))
    for k in range(1 + K):
        print(i, k, K)

        chunk = X_finite[k * chunk_size:(k + 1) * chunk_size]
        pred, pred_var = gp.predict(y, x_for_gp(X[chunk]), return_var=True)
        gpm_predictions[chunk, 2*i:2*(i + 1)] = np.vstack([pred, pred_var]).T


# Calculate log-likelihoods.


# Calculate LL ratios.
y = data["rv_single_epoch_scatter"]


def normal_lpdf(y, mu, sigma):
    ivar = sigma**(-2)
    return 0.5 * (np.log(ivar) - np.log(2 * np.pi) - (y - mu)**2 * ivar)

mu = gpm_predictions[:, 0]
ivar = gpm_predictions[:, 2]**(-2)
gpm_predictions[:, -3] = 0.5 * (np.log(ivar) - np.log(2*np.pi) - (y - mu)**2 * ivar)


def lognormal_lpdf(y, mu, sigma):
    ivar = sigma**(-2)
    return - 0.5 * np.log(2 * np.pi) - np.log(y * sigma) \
           - 0.5 * (np.log(y) - mu)**2 * ivar


mu = gpm_predictions[:, 4]
sigma = gpm_predictions[:, 6]
ivar = sigma**(-2)

gpm_predictions[:, -2] = -0.5 * np.log(2 * np.pi) - np.log(y * sigma) \
                         -0.5 * (np.log(y) - mu)**2 * ivar

# Calculate log-L ratio single/binary

ln_likelihoods = np.vstack([gpm_predictions[:, -3], gpm_predictions[:, -2]]).T
ln_likelihood = logsumexp(ln_likelihoods, axis=1)

with np.errstate(under="ignore"):
    log_tau = ln_likelihoods - ln_likelihood[:, np.newaxis]

tau_single, tau_multiple = np.exp(log_tau).T

gpm_predictions[:, -1] = tau_single

with open("tmp_gp.pkl", "wb") as fp:
    pickle.dump(gpm_predictions, fp, -1)

from astropy.table import Table
t = Table.read("data/gaia-sources-for-npm-colsubset.fits")

names = ("mu_single", "mu_single_var",
         "sigma_single", "sigma_single_var",
         "mu_multiple", "mu_multiple_var",
         "sigma_multiple", "sigma_multiple_var",
         "ln_likelihood_single", "ln_likelihood_multiple",
         "tau_single")

for i, name in enumerate(names):
    t["rv_{}".format(name)] = gpm_predictions[:, i]

t.write("results/gp-npm-rv.fits")
