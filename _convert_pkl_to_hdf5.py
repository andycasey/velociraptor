
"""
Convert pickle file to HDF5 + yaml.
"""

import h5py
import sys
import pickle
import yaml


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_prefix = sys.argv[2]

    with open(input_path, "rb") as fp:
        results = pickle.load(fp)

    # Immediately convert to h5.
    h = h5py.File(f"{output_prefix}.hdf5", "w")

    group = h.create_group("models")
    #group.attrs.update(results["config"])

    for model_name in results["models"].keys():

        sub_group = group.create_group(model_name)

        dataset_names = (
            "data_indices", 
            "mixture_model_results", 
            "gp_parameters", 
            "gp_predictions"
        )
        for i, dataset_name in enumerate(dataset_names):
            d = sub_group.create_dataset(dataset_name, 
                                         data=results["models"][model_name][i])
            
    h.close()

    with open(f"{output_prefix}.meta", "w") as fp:
        fp.write(yaml.dump(results["config"]))

    print(f"Created {output_prefix}.hdf5 and {output_prefix}.meta")

    del results
