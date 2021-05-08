import json
import numpy as np
from multimatch import Multimatch
from cumulative_performance import Cumulative_performance
from os import listdir, path

results_dir  = '../Results/'
datasets_dir = '../Datasets/'

dataset_results_dirs = listdir(results_dir)
for dataset in dataset_results_dirs:
    dataset_name = dataset.split('_')[0]
    human_scanpaths_dir = datasets_dir + dataset_name + '/human_scanpaths/'
    dataset_results_dir = results_dir + dataset + '/'

    # Esto hay que levantarlo del JSON de configuración de cada dataset
    if dataset_name == 'cIBS':
        max_scanpath_length = 16
    else:
        max_scanpath_length = 31
    if dataset_name == 'cIBS':
        number_of_images = 134
    else:
        number_of_images = 240

    # Compute human subjects metrics
    multimatch = Multimatch(dataset_name, human_scanpaths_dir, dataset_results_dir)
    multimatch.load_human_mean_per_image()

    subjects_cumulative_performance = Cumulative_performance(dataset_name, number_of_images, max_scanpath_length)
    subjects_cumulative_performance.add_human_mean(human_scanpaths_dir)

    # Compute models metrics and compare them with human subjects metrics
    models = listdir(dataset_results_dir)
    for model_name in models:
        if not(path.isdir(path.join(dataset_results_dir, model_name))):
            continue

        model_scanpaths_file = dataset_results_dir + model_name + '/Scanpaths.json'
        with open(model_scanpaths_file, 'r') as fp:
            model_scanpaths = json.load(fp)
        
        subjects_cumulative_performance.add_model(model_name, model_scanpaths)
        multimatch.add_model_vs_humans_mean_per_image(model_name, model_scanpaths)
    
    subjects_cumulative_performance.plot(save_path=dataset_results_dir)
    multimatch.plot(save_path=dataset_results_dir)