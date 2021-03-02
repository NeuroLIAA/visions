import json
import numpy as np
import multimatch
from os import listdir

results_dir  = '../Results/'
datasets_dir = '../Datasets/'
models = listdir(results_dir)

# TODO: Iterar según dataset y no por modelo
for model in models:
    model_results_path = results_dir + model + '/'

    datasets_results_dirs = listdir(model_results_path)
    # Load model's output (scanpaths) for each dataset
    for dataset_result_dir in datasets_results_dirs:
        model_scanpaths_file = model_results_path + dataset_result_dir + '/Scanpaths.json'
        with open(model_scanpaths_file) as fp:
            model_scanpaths = json.load(fp)
        
        # Get the dataset's scanpaths
        dataset_name = dataset_result_dir.split('_')[0]
        dataset_scanpaths_dir = datasets_dir + dataset_name + '/human_scanpaths/'

        multimatch.compute_averages(model_scanpaths, dataset_scanpaths_dir, results_dir)