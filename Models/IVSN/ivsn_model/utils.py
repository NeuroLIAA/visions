import json
import shutil
import pandas as pd
import numpy as np
from os import path, makedirs, listdir, pardir
from Metrics.scripts import human_scanpath_prediction
from math import floor

def rescale_coordinate(value, old_size, new_size):
    return floor((value / old_size) * new_size)

def get_human_scanpath_for_trial(human_scanpaths, image_name):
    if not human_scanpaths:
        return {}
    else:
        return human_scanpaths[image_name]

def keep_human_trials(human_scanpaths, trials_properties):
    human_trials_properties = []
    for trial in trials_properties:
        if trial['image'] in human_scanpaths:
            human_trials_properties.append(trial)
    
    if not human_trials_properties:
        raise ValueError('Human subject does not have any scanpaths for the images specified')

    return human_trials_properties

def save_scanpath_prediction_metrics(subject_scanpath, scanpath_length, image_name, output_path):
    probability_maps_path = path.join(output_path, path.join('probability_maps', image_name[:-4]))
    probability_maps = listdir(probability_maps_path)

    subject_fixations_x = np.array(subject_scanpath['X'], dtype=int)
    subject_fixations_y = np.array(subject_scanpath['Y'], dtype=int)

    image_rocs, image_nss, image_igs = [], [], []
    for index in range(1, scanpath_length):
        probability_map = pd.read_csv(path.join(probability_maps_path, 'fixation_' + str(index) + '.csv'))
        roc, nss, ig = human_scanpath_prediction.compute_metrics(probability_map, subject_fixations_y[:index], subject_fixations_x[:index])
        image_rocs.append(roc)
        image_nss.append(nss)
        image_igs.append(ig)

    subject   = path.basename(output_path)
    file_path = path.join(path.join(output_path, pardir), subject + '_results.json')
    if path.exists(file_path):
        model_subject_metrics = load_dict_from_json(file_path)
    else:
        model_subject_metrics = {}
    
    model_subject_metrics[image_name] = {'AUC': np.mean(image_rocs), 'NSS': np.mean(image_nss), 'IG': np.mean(image_igs)}  
    save_to_json(file_path, model_subject_metrics)

    # Clean up
    shutil.rmtree(probability_maps_path)

def save_probability_map(fixation_number, image_name, probability_map, output_path):
    save_path = path.join(output_path, path.join('probability_maps', image_name[:-4]))
    if not path.exists(save_path):
        makedirs(save_path)

    probability_map_df = pd.DataFrame(probability_map)
    probability_map_df.to_csv(path.join(save_path, 'fixation_' + str(fixation_number + 1) + '.csv'))

def load_human_scanpaths(human_scanpaths_dir, human_subject):
    if human_subject is None:
        return {}

    human_scanpaths_files = listdir(human_scanpaths_dir)
    human_subject_str     = str(human_subject)
    if human_subject < 10: human_subject_str = '0' + human_subject_str
    human_subject_file    = 'subj' + human_subject_str + '_scanpaths.json'
    if not human_subject_file in human_scanpaths_files:
        raise NameError('Scanpaths for human subject ' + human_subject_str + ' not found!')
    
    human_scanpaths = load_dict_from_json(path.join(human_scanpaths_dir, human_subject_file))

    # Convert to int
    for trial in human_scanpaths:
        scanpath = human_scanpaths[trial]
        scanpath['X'] = [int(x_coord) for x_coord in scanpath['X']]
        scanpath['Y'] = [int(y_coord) for y_coord in scanpath['Y']]

    return human_scanpaths  

def load_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)

def load_from_dataset(dataset_path, filename):
    return load_dict_from_json(path.join(dataset_path, filename))

def save_scanpaths(output_path, scanpaths, filename='Scanpaths.json'):
    if not path.exists(output_path):
        makedirs(output_path)
    
    save_to_json(path.join(output_path, filename), scanpaths)

def save_to_json(file, data):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)