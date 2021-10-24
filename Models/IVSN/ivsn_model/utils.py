import json
import shutil
import pandas as pd
import numpy as np
from os import path, makedirs, listdir
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