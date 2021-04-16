import json
import numpy as np
import multimatch
from cumulative_performance import Cumulative_performance
from os import listdir, path

results_dir  = '../Results/'
datasets_dir = '../Datasets/'

dataset_results_dirs = listdir(results_dir)
for dataset in dataset_results_dirs:
    dataset_name = dataset.split('_')[0]
    human_scanpaths_dir = datasets_dir + dataset_name + '/human_scanpaths/'
    dataset_result_dir = results_dir + dataset + '/'

    # Esto hay que levantarlo del JSON de configuración de cada dataset
    if dataset_name == 'cIBS':
        max_scanpath_length = 16
    else:
        max_scanpath_length = 31

    # Compute human subjects metrics
    mm_humans = multimatch.human_average_per_image(human_scanpaths_dir, dataset_result_dir)

    subjects_cumulative_performance = Cumulative_performance(dataset_name, max_scanpath_length)
    subjects_cumulative_performance.add_human_average(human_scanpaths_dir)

    # Compute models metrics and compare them with human subjects metrics
    models = listdir(dataset_result_dir)
    for model_name in models:
        if not(path.isdir(path.join(dataset_result_dir, model_name))):
            continue

        model_scanpaths_file = dataset_result_dir + model_name + '/Scanpaths.json'
        with open(model_scanpaths_file, 'r') as fp:
            model_scanpaths = json.load(fp)
        
        subjects_cumulative_performance.add_model(model_name, model_scanpaths)

        mm_model_vs_humans = multimatch.model_vs_humans_average_per_image(model_scanpaths, human_scanpaths_dir)

        multimatch.plot(model_name, dataset_name, mm_model_vs_humans, mm_humans)
    
    subjects_cumulative_performance.plot()