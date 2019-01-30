"""
Convert results files.
"""


import os
import sys
import pickle
import yaml

if __name__ == "__main__":

    output_path = sys.argv[1]
    joint_config_path = sys.argv[2]
    result_paths = sys.argv[3:]

    with open(joint_config_path, "r") as fp:
        config = yaml.load(fp)

    output = dict(config=config)

    for i, result_path in enumerate(result_paths):

        model_name = os.path.basename(result_path).split(".")[0]

        with open(result_path, "rb") as fp:
            results = pickle.load(fp)

        output[model_name] = [
            results["results"], 
            results["gp_parameters"], 
            results["gp_predictions"]
        ]
        

    with open(output_path, "wb") as fp:
        pickle.dump(output, fp)
