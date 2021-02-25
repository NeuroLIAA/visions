import multimatch_gaze as mm
import json
import numpy as np
import pandas as pd
from os import listdir

def compute_averages(model_scanpaths, dataset_scanpaths_dir, results_dir):
    multimatch_humans_average_per_image = compute_human_average_per_image(dataset_scanpaths_dir)
    print(multimatch_humans_average_per_image)

def compute_human_average_per_image(dataset_scanpaths_dir):
    " Returns dictionary with image names as keys and multimatch arrays as values "
    subjects_scanpaths_files = listdir(dataset_scanpaths_dir)
    Multimatch_values_per_image = {}

    # Compute multimatch for each image for every pair of subjects
    for left_subject_scanpaths_file, right_subject_scanpaths_file in pairwise(subjects_scanpaths_files):
        with open(dataset_scanpaths_dir + left_subject_scanpaths_file, 'r') as fp:
            left_subject_scanpaths = json.load(fp)

        # Use pandas to index by image name
        right_subject_scanpaths_df = pd.read_json(dataset_scanpaths_dir + right_subject_scanpaths_file)
        
        for trial_index in range(len(left_subject_scanpaths)):
            left_subject_trial_info = left_subject_scanpaths[trial_index]
            target_found = bool(left_subject_trial_info['target_found'])
            # Only scanpaths where the target was found are taken into account
            if not(target_found):
                continue

            image_name = left_subject_trial_info['image']
            right_subject_trial_info = right_subject_scanpaths_df.loc[right_subject_scanpaths_df['image'] == image_name]
            if right_subject_trial_info.empty:
                continue
            if not(bool(right_subject_trial_info['target_found'].values[0])):
                continue

            screen_size = [left_subject_trial_info['image_height'], left_subject_trial_info['image_width']]

            # Al computar con aquellos de los modelos, agregar if para reescalar el scanpath del otro sujeto si difiere en el tamaño de la imagen (Zhang usa 1028x1280 para el modelo y 1024x1280 para los humanos)

            left_subject_scanpath_X = left_subject_trial_info['X']
            left_subject_scanpath_Y = left_subject_trial_info['Y']
            left_subject_scanpath_time = [t * 0.0001 for t in left_subject_trial_info['T']]

            right_subject_scanpath_X = right_subject_trial_info['X'].values[0]
            right_subject_scanpath_Y = right_subject_trial_info['Y'].values[0]
            right_subject_scanpath_time = [t * 0.0001 for t in right_subject_trial_info['T'].values[0]]

            left_subject_scanpath  = np.array(list(zip(left_subject_scanpath_X, left_subject_scanpath_Y, left_subject_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
            right_subject_scanpath = np.array(list(zip(right_subject_scanpath_X, right_subject_scanpath_Y, right_subject_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])

            trial_multimatch_result = mm.docomparison(left_subject_scanpath, right_subject_scanpath, screen_size)

            if image_name in Multimatch_values_per_image:
                trial_multimatch_average = Multimatch_values_per_image[image_name]
                Multimatch_values_per_image[image_name] = np.add(trial_multimatch_average, trial_multimatch_result) / 2
            else:
                Multimatch_values_per_image[image_name] = trial_multimatch_result
    
    return Multimatch_values_per_image

def pairwise(it):
    it = iter(it)
    while True:
        try:
            yield next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return