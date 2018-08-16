
"""
A post-processing Gaussian Process model for radial velocity component parameters
using a subset of the data.
"""

import itertools # dat feel
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.optimize as op
import yaml
from astropy.io import fits
from matplotlib.ticker import MaxNLocator
from time import time

import george
import george.kernels

from mpl_utils import mpl_style


seed = 123

np.random.seed(seed) # for reproducibility
matplotlib.style.use(mpl_style) # for style


with open("npm-config.rv.yaml", "r") as fp:
    config = yaml.load(fp)


data = fits.open(config["data_path"])[1].data

# Load results.
with open("results/rv_single_epoch_scatter.v3.pkl", "rb") as fp:
    npm_results = pickle.load(fp)


# Take a random M results to fit the GP
M = 1000

# And predict a random K stars.
cross_validate = True
K = 5000

npm_index = 2

N, D = npm_results.shape
finite_indices = np.arange(N)[np.isfinite(npm_results[:, 0])]
indices = np.random.choice(finite_indices, M, replace=False)


# Plot these.
scatter_kwds = dict(s=1, alpha=0.5, cmap="viridis")

fig, ax = plt.subplots()
ax.scatter(data["bp_rp"][indices],
           data["absolute_rp_mag"][indices],
           c=npm_results[:, npm_index][indices], **scatter_kwds)

ax.set_ylim(ax.get_ylim()[::-1])




# Now let's fit a Gaussian Process to one of the variables.
x = np.vstack([data[ln][indices] for ln in config["kdtree_label_names"]]).T
y = npm_results[indices, npm_index]


metric = np.var(x, axis=0)


#kernel = np.var(y) * george.kernels.Matern32Kernel(metric, ndim=x.shape[1])
kernel = np.var(y) * george.kernels.ExpKernel(metric, ndim=x.shape[1]) # 0.22 std for 
#kernel = np.var(y) * george.kernels.RationalQuadraticKernel(metric=1.0, log_alpha=-1, ndim=x.shape[1]) --> 0.20 std
#kernel = np.var(y) * george.kernels.ExpSquaredKernel(metric=1.0, ndim=x.shape[1])

#kernel = 0.5 * np.var(y) * (george.kernels.Matern32Kernel(metric, ndim=x.shape[1]) \
#                        +   george.kernels.ExpKernel(metric, ndim=x.shape[1]))


kernel = george.kernels.ExpKernel(metric, ndim=x.shape[1]) \
       + george.kernels.Matern32Kernel(metric, ndim=x.shape[1])


# Now fit the hyperparameters.
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True, fit_white_noise=True)
#solver=george.HODLRSolver, seed=seed)

def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

# Initialize
#xgp = np.copy(x)
#xgp[:,0] = 1.0/xgp[:, 0]
xgp = x.copy()
xgp[:,0] = 1.0/x[:,0]
gp.compute(xgp)
print("Initial \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
print("initial \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

p0 = gp.get_parameter_vector()

t_init = time()
result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
t_opt = time() - t_init

gp.set_parameter_vector(result.x)
print("Result: {}".format(result))
print("Final logL = {:.2f}".format(gp.log_likelihood(y)))
print("Took {:.0f} seconds to optimize".format(t_opt))

# Now show that model.
if cross_validate:
    idx = np.random.choice(finite_indices, min(K, finite_indices.size),
                           replace=False)
else:
    idx = indices

xp = np.vstack([data[ln][idx] for ln in config["kdtree_label_names"]]).T
#xp[:,0] = 1.0/xp[:,0]
xpgp = xp.copy()
xpgp[:, 0] = 1.0/xpgp[:, 0]

yp = npm_results[idx, npm_index]
pred, pred_var = gp.predict(y, xpgp, return_var=True)




v = np.hstack([pred, y])
kwds = scatter_kwds.copy()
kwds.update(vmin=np.min(v), vmax=np.max(v))

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = np.atleast_2d(axes).T

axes[0, 0].scatter(x.T[0], x.T[1], c=y, **kwds)
axes[0, 1].scatter(x.T[0], x.T[2], c=y, **kwds)

axes[1, 0].scatter(xp.T[0], xp.T[1], c=pred, **kwds)
axes[1, 1].scatter(xp.T[0], xp.T[2], c=pred, **kwds)


vminmax = np.std(np.abs(pred - yp))

residual_kwds = scatter_kwds.copy()
residual_kwds.update(vmin=-vminmax, vmax=+vminmax)

axes[2, 0].scatter(xp.T[0], xp.T[1], c=pred - yp, **residual_kwds)
axes[2, 1].scatter(xp.T[0], xp.T[2], c=pred - yp, **residual_kwds)

xlims = np.array([ax.get_xlim() for ax in axes.flatten()])
xlims = (np.min(xlims), np.max(xlims))

for ax_col, ylabel in zip(axes.T, ("absolute rp mag", "apparent rp mag")):
    # Common limits for each column.
    ylims = np.array([ax.get_ylim() for ax in ax_col])
    ylims = (np.min(ylims), np.max(ylims))

    for ax in ax_col:
        ax.set_ylabel(r"\textrm{{{0}}}".format(ylabel))
        ax.set_ylim(ylims)

for ax_row, descr, N in zip(axes, ("data", "model", "residual"), (M, K, K)):
    for ax in ax_row:
        ax.set_title(r"\textrm{{{0}}}".format(descr))
        ax.text(0.95, 0.95, r"${}$".format(N),
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes)


for ax in axes.flatten():
    ax.set_xlim(xlims)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel(r"\textrm{bp - rp}")


for ax in axes[:, 0]:
    ax.set_ylim(10, -10)

fig.tight_layout()


# Compare model with that measured from the non-parametric model.
latex_labels = (r"\textrm{bp - rp}", r"\textrm{absolute rp mag}",
                r"\textrm{apparent rp mag}")

fig, axes = plt.subplots(3, figsize=(9, 7.5))
for i, (ax, xlabel) in enumerate(zip(axes, latex_labels)):
    ax.scatter(xp.T[i], pred - yp, **scatter_kwds)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"\textrm{residual}")

    ax.axhline(0, c="#666666", zorder=-1, linestyle=":", lw=1)

fig.tight_layout()


fig, ax = plt.subplots()
ax.scatter(yp, pred, **scatter_kwds)
ax.errorbar(yp, pred, yerr=np.sqrt(pred), fmt=None, ecolor="#666666", alpha=0.05,
            zorder=-1)
limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
limits = (np.min(limits), np.max(limits))
ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1, lw=1)
ax.set_xlim(limits)
ax.set_ylim(limits)

ax.set_xlabel(r"\textrm{npm}")
ax.set_ylabel(r"\textrm{gpm}")

# If we have a model for the single and binary star things now,... let's just
# use that and evaluate binarity for each object.

fig, axes = plt.subplots(3)
for i, ax in enumerate(axes):

    # mean of other vals.
    xm = np.ones((50, 3)) * np.mean(xp, axis=0)
    xm[:, i] = np.linspace(np.min(xp.T[i]), np.max(xp.T[i]), 50)

    xmgp = xm.copy()
    xmgp[:, 0] = 1.0/xmgp[:, 0]
    ym, _ = gp.predict(y, xmgp)

    ax.scatter(xp.T[i], yp, s=1, c="#cccccc", alpha=0.25, zorder=-1)
    ax.plot(xm.T[i], ym, "-", lw=2, c="tab:blue")


# Fit for all
gps = []
for npm_index in range(1, 5):

    print("Working on index = {:.0f}".format(npm_index))


    # Now let's fit a Gaussian Process to one of the variables.
    x = np.vstack([data[ln][indices] for ln in config["kdtree_label_names"]]).T
    y = npm_results[indices, npm_index]


    metric = np.var(x, axis=0)


    #kernel = np.var(y) * george.kernels.Matern32Kernel(metric, ndim=x.shape[1])
    kernel = np.var(y) * george.kernels.ExpKernel(metric, ndim=x.shape[1]) # 0.22 std for 
    #kernel = np.var(y) * george.kernels.RationalQuadraticKernel(metric=1.0, log_alpha=-1, ndim=x.shape[1]) --> 0.20 std
    #kernel = np.var(y) * george.kernels.ExpSquaredKernel(metric=1.0, ndim=x.shape[1])

    #kernel = 0.5 * np.var(y) * (george.kernels.Matern32Kernel(metric, ndim=x.shape[1]) \
    #                        +   george.kernels.ExpKernel(metric, ndim=x.shape[1]))


    kernel = george.kernels.ExpKernel(metric, ndim=x.shape[1]) \
           + george.kernels.Matern32Kernel(metric, ndim=x.shape[1])


    # Now fit the hyperparameters.
    gp = george.GP(kernel, mean=np.mean(y), white_noise=10,
                   fit_mean=True, fit_white_noise=True)
    #solver=george.HODLRSolver, seed=seed)

    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    # Initialize
    #xgp = np.copy(x)
    #xgp[:,0] = 1.0/xgp[:, 0]
    xgp = x.copy()
    xgp[:,0] = 1.0/x[:,0]
    gp.compute(xgp)
    print("Initial \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
    print("initial \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

    p0 = gp.get_parameter_vector()

    t_init = time()
    result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    t_opt = time() - t_init

    gp.set_parameter_vector(result.x)
    print("Result: {}".format(result))
    print("Final logL = {:.2f}".format(gp.log_likelihood(y)))
    print("Took {:.0f} seconds to optimize".format(t_opt))

    gps.append(gp)


hps = [gp.get_parameter_dict() for gp in gps]
with open("gp-rv-hps.pkl", "wb") as fp:
    pickle.dump((hps, x, y), fp, -1)