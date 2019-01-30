
"""
Convert config format from run_analysis -> run_analysis2.py
"""

import os
import sys
import pickle
import yaml

if __name__ == "__main__":

    output_path = sys.argv[1]
    model_paths = sys.argv[2:]

    config = dict()
    for i, model_path in enumerate(model_paths):

        model_name = os.path.basename(model_path).split(".")[1]

        with open(model_path, "r") as fp:
            model_config = yaml.load(fp)

        if i == 0:
            for key in ("data_path", "results_path", "model_path", "random_seed"
                        "number_of_sources", "multiprocessing", "suppress_stan_output"):
                if key in model_config:
                    config[key] = model_config[key]


        foo = dict()
        for key in ("kdtree_label_names", "kdtree_minimum_points", 
                    "kdtree_maximum_points", "kdtree_relative_scales", 
                    "kdtree_minimum_radius", "kdtree_maximum_radius"):
            foo[key] = model_config[key]

        foo["predictor_label_name"] = model_config["predictor_label_names"][0]

        config[model_name] = foo



    with open(output_path, "w") as fp:
        fp.write(yaml.dump(config))
