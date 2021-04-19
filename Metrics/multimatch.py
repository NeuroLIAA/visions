import multimatch_gaze as mm
import json
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path

def plot(model, dataset, multimatch_values_per_image_x, multimatch_values_per_image_y):
    x_vector = []
    y_vector = []
    for image_name in multimatch_values_per_image_x.keys():
        if not(image_name in multimatch_values_per_image_y):
            continue

        shape_value_x = np.mean(multimatch_values_per_image_x[image_name][:-1])
        shape_value_y = np.mean(multimatch_values_per_image_y[image_name][:-1])

        x_vector.append(shape_value_x)
        y_vector.append(shape_value_y)
    
    fig, ax = plt.subplots()
    ax.scatter(x_vector, y_vector, color='red', alpha=0.5)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, linestyle='dashed', c='.3')
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    plt.xlabel('Model vs human average multimatch')
    plt.ylabel('Human average multimatch')
    min_coord = min(min(ax.get_xlim()), min(ax.get_ylim()))
    max_coord = max(max(ax.get_xlim()), max(ax.get_ylim()))
    #plt.xlim(min_coord, max_coord)
    #plt.ylim(min_coord, max_coord)
    plt.title(model + ' (' + dataset + ' dataset)')
    plt.show()

def model_vs_humans_average_per_image(model_scanpaths, dataset_scanpaths_dir):
    " For each scanpath produced by the model, multimatch is calculated against the scanpath of that same image for every human subject"
    " The average is computed for each image"
    " Output is a dictionary where the image names are the keys and the multimatch averages are the values"
    multimatch_model_vs_humans_average_per_image = {}

    total_values_per_image = {}
    subjects_scanpaths_files = listdir(dataset_scanpaths_dir)
    for subject_filename in subjects_scanpaths_files:
        with open(dataset_scanpaths_dir + subject_filename, 'r') as fp:
            subject_scanpaths = json.load(fp)
        for image_name in model_scanpaths.keys():
            if not(image_name in subject_scanpaths):
                continue

            model_trial_info = model_scanpaths[image_name]
            subject_trial_info = subject_scanpaths[image_name]

            trial_multimatch_result = compute_multimatch(subject_trial_info, model_trial_info)

            # Check if result is empty
            if not trial_multimatch_result:
                continue

            if image_name in multimatch_model_vs_humans_average_per_image:
                multimatch_trial_value_acum = multimatch_model_vs_humans_average_per_image[image_name]
                multimatch_model_vs_humans_average_per_image[image_name] = np.add(multimatch_trial_value_acum, trial_multimatch_result)
                total_values_per_image[image_name] += 1 
            else:
                multimatch_model_vs_humans_average_per_image[image_name] = trial_multimatch_result
                total_values_per_image[image_name] = 1

    # Compute average per image
    for image_name in multimatch_model_vs_humans_average_per_image.keys():
        multimatch_model_vs_humans_average_per_image[image_name] = (np.divide(multimatch_model_vs_humans_average_per_image[image_name], total_values_per_image[image_name])).tolist()
    
    return multimatch_model_vs_humans_average_per_image

def human_average_per_image(dataset_scanpaths_dir, dataset_result_dir):
    " For each human subject, multimatch is computed against every other human subject, for each trial"
    " The average is computed for each trial (i.e. for each image)"
    " Output is a dictionary where the image names are the keys and the multimatch averages are the values"
    multimatch_human_average_per_image = {}
    # Check if it was already computed
    multimatch_human_average_json_file = dataset_result_dir + 'multimatch_human_average_per_image.json'
    if path.exists(multimatch_human_average_json_file):
        with open(multimatch_human_average_json_file, 'r') as fp:
            multimatch_human_average_per_image = json.load(fp)
    else:
        total_values_per_image = {}
        # Compute multimatch for each image for every pair of subjects
        subjects_scanpaths_files = listdir(dataset_scanpaths_dir)
        for subject_filename in list(subjects_scanpaths_files):
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

                    trial_multimatch_result = compute_multimatch(subject_trial_info, subject_to_compare_trial_info)

                    # Check if result is empty
                    if not trial_multimatch_result:
                        continue

                    if image_name in multimatch_human_average_per_image:
                        multimatch_trial_value_acum = multimatch_human_average_per_image[image_name]
                        multimatch_human_average_per_image[image_name] = np.add(multimatch_trial_value_acum, trial_multimatch_result)
                        total_values_per_image[image_name] += 1 
                    else:
                        multimatch_human_average_per_image[image_name] = trial_multimatch_result
                        total_values_per_image[image_name] = 1

        # Compute average per image
        for image_name in multimatch_human_average_per_image.keys():
            multimatch_human_average_per_image[image_name] = (np.divide(multimatch_human_average_per_image[image_name], total_values_per_image[image_name])).tolist()
        
        with open(multimatch_human_average_json_file, 'w') as fp:
            json.dump(multimatch_human_average_per_image, fp, indent = 4)

    return multimatch_human_average_per_image

def compute_multimatch(trial_info, trial_to_compare_info):
    target_found = trial_info['target_found'] and trial_to_compare_info['target_found']
    if not(target_found):
        return []

    screen_size = [trial_info['image_width'], trial_info['image_height']]

    trial_scanpath_X = trial_info['X']
    trial_scanpath_Y = trial_info['Y']
    trial_scanpath_length = len(trial_scanpath_X)
    trial_scanpath_time = get_scanpath_time(trial_info, trial_scanpath_length)

    trial_to_compare_image_width  = trial_to_compare_info['image_width']
    trial_to_compare_image_height = trial_to_compare_info['image_height']

    trial_to_compare_scanpath_X = trial_to_compare_info['X']
    trial_to_compare_scanpath_Y = trial_to_compare_info['Y']
    trial_to_compare_scanpath_length = len(trial_to_compare_scanpath_X)
    trial_to_compare_scanpath_time = get_scanpath_time(trial_to_compare_info, trial_to_compare_scanpath_length)

    # Rescale accordingly
    trial_to_compare_scanpath_X = [rescale_coordinate(x, trial_to_compare_image_width, screen_size[0]) for x in trial_to_compare_scanpath_X]
    trial_to_compare_scanpath_Y = [rescale_coordinate(y, trial_to_compare_image_height, screen_size[1]) for y in trial_to_compare_scanpath_Y]

    # Multimatch can't be computed for scanpaths with length shorter than 3
    if trial_scanpath_length < 3 or trial_to_compare_scanpath_length < 3:
        return []

    trial_scanpath = np.array(list(zip(trial_scanpath_X, trial_scanpath_Y, trial_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
    trial_to_compare_scanpath = np.array(list(zip(trial_to_compare_scanpath_X, trial_to_compare_scanpath_Y, trial_to_compare_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])

    return mm.docomparison(trial_scanpath, trial_to_compare_scanpath, screen_size)

def get_scanpath_time(trial_info, length):
    if 'T' in trial_info:
        scanpath_time = [t * 0.0001 for t in trial_info['T']]
    else:
        # Dummy
        scanpath_time = [0.3] * length
    
    return scanpath_time

def rescale_coordinate(value, old_size, new_size):
    return (value / old_size) * new_size