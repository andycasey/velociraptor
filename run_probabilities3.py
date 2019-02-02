
"""
Calculate probability distributions.
"""

import h5py
import logging
import multiprocessing as mp
import numpy as np
import sys
import pickle
import yaml
import warnings
from tqdm import tqdm
from collections import OrderedDict
from astropy.io import fits
from scipy import (special, stats)

import npm_utils as npm


if __name__ == "__main__":

    data_path = sys.argv[1]
    meta_path = sys.argv[2]
    
    if not data_path.endswith(".hdf5"):
        raise ValueError("data_path should be a .hdf5 file")
    if not meta_path.endswith(".meta"):
        raise ValueError("meta_path should be a .meta file")

    # Load meta.
    with open(meta_path, "r") as fp:
        meta = yaml.load(fp)

    # Load data.
    data = h5py.File(data_path, "r+")

    model_names = list(data["models"].keys())
    predictor_label_names = [meta["models"][mn]["predictor_label_name"] \
                             for mn in model_names]

    all_label_names = [] + predictor_label_names
    for model_name, model_config in meta["models"].items():
        all_label_names.extend(model_config["kdtree_label_names"])

    all_label_names = list(np.unique(all_label_names))

    # Load the predictors that we need.
    with fits.open(meta["data_path"]) as image:
    
        finite = np.all([np.isfinite(image[1].data[ln]) for ln in all_label_names], axis=0)
        Y = np.array([image[1].data[pln] for pln in predictor_label_names]).T[finite]

        # Store relevant data.
        for label_name in all_label_names + ["source_id"]:
            if label_name not in data:
                data.create_dataset(label_name, data=np.array(image[1].data[label_name])[finite])


    data.close()
    del data

    K = meta.get("number_of_draws", 10)
    N, M = Y.shape

    for m, model_name in enumerate(model_names[1:]):

        print(f"Calculating probabilities for {model_name} with K = {K}")


        def _init_worker(gp_predictions_, y_, rhos_, bounds_):
            global gp_predictions, y, rhos, bounds
            gp_predictions, y, rhos, bounds = (gp_predictions_, y_, rhos_, bounds_)
            return None


        def _get_lnprob(n):

            gp_predictions_, y_ = gp_predictions[n], y[n]

            if not np.all(np.isfinite(gp_predictions_)) or not np.isfinite(y_):
                return (n, -1e25, 0.5)

            mu = gp_predictions_[::2]
            diag = np.atleast_2d(gp_predictions_[1::2])**0.5

            cov = diag * rhos * diag.T

            try:
                draws = np.random.multivariate_normal(mu, cov, K).T

            except ValueError:
                print(f"ERROR ON {n}, {mu}, {cov}, {diag}, {rhos}")
                return (n, -1e25, 0.5)

            # Clip to bounded values.
            parameter_names = ("theta", 
                               "mu_single", "sigma_single", 
                               "mu_multiple", "sigma_multiple")

            for i, parameter_name in enumerate(parameter_names):
                try:
                    l, u = bounds[parameter_name]

                except KeyError:
                    continue

                else:
                    draws[i] = np.clip(draws[i], l, u)

            draws_3_min = np.log(draws[1] + 5 * draws[2]) + draws[4]**2
            draws[3] = np.max([draws[3], draws_3_min], axis=0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                lnprob_ = np.vstack([
                    np.log(draws[0]) + npm.normal_lpdf(y_, draws[1], draws[2]),
                    np.log(1 - draws[0]) + npm.lognormal_lpdf(y_, draws[3], draws[4])
                ]).T

                lnprob_[~np.isfinite(lnprob_)] = -1e25

                p_single_ = np.exp(lnprob_[:, 0] - special.logsumexp(lnprob_, axis=1))

            # TODO HACK
            is_def_single = y_ < draws[1]
            p_single_[is_def_single] = 1.0

            return (n, lnprob_, p_single_)


        chunk_size = 1000000
        chunks = int(np.ceil(N/chunk_size))

        print(f"Using chunk size of {chunk_size} (expect {chunks} chunks)")

        with tqdm(total=N) as pbar:

            for i in range(chunks):
                start, end = (chunk_size * i, chunk_size * (i + 1))

                with h5py.File(data_path, "r+") as data:

                    group = data["models"][model_name]

                    if i == 0:
                        if "lnprob" not in group:
                            group.create_dataset("lnprob", shape=(N, K, 2), dtype=float)

                        if "p_single" not in group:
                            group.create_dataset("p_single", shape=(N, K), dtype=float)

                        if "classification" not in group:
                            group.create_dataset("classification", shape=(N, ), dtype=int)

                        if "confidence" not in group:
                            group.create_dataset("confidence", shape=(N, ), dtype=float)

                    initargs = (
                        group["gp_predictions"],
                        Y[:, m],
                        np.corrcoef(np.transpose(group["mixture_model_results"])),
                        meta["models"][model_name]["bounds"],
                    )

                    C = min(N - chunk_size * i, chunk_size)
                    end = start + C
                    lnprob_tmp = np.memmap(
                                "lnprob.tmp", 
                                mode="w+", dtype=float, 
                                shape=(C, K, 2))

                    p_single_tmp = np.memmap(
                                "p_single.tmp",
                                mode="w+", dtype=float,
                                shape=(C, K))

                    with mp.Pool(initializer=_init_worker, initargs=initargs) as p:
                        for n, lnp, ps in p.imap_unordered(_get_lnprob, range(start, end)):
                            lnprob_tmp[n - start] = lnp
                            p_single_tmp[n - start] = ps
                            pbar.update(1)

                    lnprob_tmp.flush()
                    p_single_tmp.flush()

                    confidence = np.sum(p_single_tmp > 0.5, axis=1)/K
                    assert confidence.size == C
                    group["lnprob"][start:end] = lnprob_tmp
                    group["p_single"][start:end] = p_single_tmp
                    group["classification"][start:end] = np.round(confidence).astype(int)
                    group["confidence"][start:end] = confidence

                    del lnprob_tmp, p_single_tmp, confidence


    chunk_size = 100000
    chunks = int(np.ceil(N/chunk_size))

    # Calculate joint probabilities.
    print(f"Calculating joint probabilities")

    with tqdm(total=N) as pbar:
        for i in range(chunks):
            start, end = (chunk_size * i, chunk_size * (i + 1))

            C = min(N - chunk_size * i, chunk_size)
            end = start + C

            with h5py.File(data_path, "r+") as data:

                if i == 0:
                    if "joint_p_single" not in data:
                        data.create_dataset("joint_p_single", shape=(N, K), dtype=float)

                    if "joint_classification" not in data:
                        data.create_dataset("joint_classification", shape=(N,), dtype=int)

                    if "joint_confidence" not in data:
                        data.create_dataset("joint_confidence", shape=(N, ), dtype=float)



                # Get lnprobs.
                lnprobs = np.sum(
                    [data["models"][model_name]["lnprob"][start:end] \
                    for model_name in model_names], axis=0)

                joint_p_single = np.exp(lnprobs[:, :, 0] - special.logsumexp(lnprobs, axis=-1))
                data["joint_p_single"][start:end] = joint_p_single

                confidence = np.sum(joint_p_single > 0.5, axis=1)/K
                data["joint_classification"][start:end] = np.round(confidence).astype(int)

                data["joint_confidence"][start:end] = confidence

                pbar.update(C)