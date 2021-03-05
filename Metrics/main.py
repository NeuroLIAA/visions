import json
import numpy as np
import multimatch
from os import listdir

results_dir  = '../Results/'
datasets_dir = '../Datasets/'

dataset_results_dirs = listdir(results_dir)
for dataset in dataset_results_dirs:
    dataset_name = dataset.split('_')[0]
    dataset_scanpaths_dir = datasets_dir + dataset_name + '/human_scanpaths/'
    dataset_result_dir = results_dir + dataset + '/'

    mm_humans = multimatch.compute_human_average_per_image(dataset_scanpaths_dir, dataset_result_dir)

# for model in models:
#     model_results_path = results_dir + model + '/'
#     # Load model's output (scanpaths) for each dataset
#     for dataset_result_dir in datasets_results_dirs:
#         model_scanpaths_file = model_results_path + dataset_result_dir + '/Scanpaths.json'
#         with open(model_scanpaths_file) as fp:
#             model_scanpaths = json.load(fp)