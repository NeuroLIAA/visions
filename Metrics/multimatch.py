import multimatch_gaze as mm
import json
import numpy as np
from os import listdir

def compute_averages(model_scanpaths, dataset_scanpaths_dir, results_dir):
    multimatch_humans_average_per_image = compute_human_average_per_image(dataset_scanpaths_dir)
    print(multimatch_humans_average_per_image)

def compute_human_average_per_image(dataset_scanpaths_dir):
    " Returns dictionary with image names as keys and multimatch arrays as values "
    subjects_scanpaths_files = listdir(dataset_scanpaths_dir)
    Multimatch_values_per_image = {}
    total_values_per_image = {}
    # Compute multimatch for each image for every pair of subjects
    for subject_filename in subjects_scanpaths_files:
        subjects_scanpaths_files.remove(subject_filename) 
        with open(dataset_scanpaths_dir + subject_filename, 'r') as fp:
            subject_scanpaths = json.load(fp)
        for subject_to_compare_filename in subjects_scanpaths_files:
            with open(dataset_scanpaths_dir + subject_to_compare_filename, 'r') as fp:
                subject_to_compare_scanpaths = json.load(fp)
            for image_name in subject_scanpaths.keys():
                if not(image_name in subject_to_compare_scanpaths):
                    continue

                subject_trial_info = subject_scanpaths[image_name]
                subject_to_compare_trial_info = subject_to_compare_scanpaths[image_name]

                # Only scanpaths where the target was found are taken into account
                target_found = bool(subject_trial_info['target_found']) and bool(subject_to_compare_trial_info['target_found'])
                if not(target_found):
                    continue

                screen_size = [subject_trial_info['image_width'], subject_trial_info['image_height']]

                # Al computar con aquellos de los modelos, agregar if para reescalar el scanpath del otro sujeto si difiere en el tamaño de la imagen (Zhang usa 1028x1280 para el modelo y 1024x1280 para los humanos)

                subject_scanpath_X = subject_trial_info['X']
                subject_scanpath_Y = subject_trial_info['Y']
                subject_scanpath_time = [t * 0.0001 for t in subject_trial_info['T']]

                subject_to_compare_scanpath_X = subject_to_compare_trial_info['X']
                subject_to_compare_scanpath_Y = subject_to_compare_trial_info['Y']
                subject_to_compare_scanpath_time = [t * 0.0001 for t in subject_to_compare_trial_info['T']]

                subject_scanpath = np.array(list(zip(subject_scanpath_X, subject_scanpath_Y, subject_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
                subject_to_compare_scanpath = np.array(list(zip(subject_to_compare_scanpath_X, subject_to_compare_scanpath_Y, subject_to_compare_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])

                if (image_name == 'img212.jpg'):
                    print('Subject 1:' + subject_filename)
                    print('Subject 2: ' + subject_to_compare_filename)
                    print('Scanpath 1: ' + str(subject_scanpath))
                    print('Scanpath 2: ' + str(subject_to_compare_scanpath))

                trial_multimatch_result = mm.docomparison(subject_scanpath, subject_to_compare_scanpath, screen_size)

                if (image_name == 'img212.jpg'):
                    print('Multimatch result: ' + str(trial_multimatch_result))

                if image_name in Multimatch_values_per_image:
                    multimatch_trial_value_acum = Multimatch_values_per_image[image_name]
                    Multimatch_values_per_image[image_name] = np.add(multimatch_trial_value_acum, trial_multimatch_result)
                    total_values_per_image[image_name] += 1 
                else:
                    Multimatch_values_per_image[image_name] = trial_multimatch_result
                    total_values_per_image[image_name] = 1

    # Compute average per image
    for image_name in Multimatch_values_per_image.keys():
        Multimatch_values_per_image[image_name] = Multimatch_values_per_image[image_name] / total_values_per_image[image_name]

    return Multimatch_values_per_image
