import json
from os import path
from math import floor

def rescale_coordinate(value, old_size, new_size):
    return floor((value / old_size) * new_size)

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