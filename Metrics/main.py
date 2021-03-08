import json
import numpy as np
import multimatch
from os import listdir, path

results_dir  = '../Results/'
datasets_dir = '../Datasets/'

dataset_results_dirs = listdir(results_dir)
for dataset in dataset_results_dirs:
    dataset_name = dataset.split('_')[0]
    dataset_scanpaths_dir = datasets_dir + dataset_name + '/human_scanpaths/'
    dataset_result_dir = results_dir + dataset + '/'

    mm_humans = multimatch.human_average_per_image(dataset_scanpaths_dir, dataset_result_dir)

    models = listdir(dataset_result_dir)
    for model in models:
        if not(path.isdir(path.join(dataset_result_dir, model))):
            continue

        model_scanpaths_file = dataset_result_dir + model + '/Scanpaths.json'
        with open(model_scanpaths_file, 'r') as fp:
            model_scanpaths = json.load(fp)
        
        mm_model_vs_humans = multimatch.model_vs_humans_average_per_image(model_scanpaths, dataset_scanpaths_dir)

        multimatch.plot(model, dataset_name, mm_model_vs_humans, mm_humans)