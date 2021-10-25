import json
import random
import re
from os import listdir, path, scandir
from .. import constants

def dir_is_too_heavy(path):
    nmbytes = sum(d.stat().st_size for d in scandir(path) if d.is_file()) / 2**20
    
    return nmbytes > constants.MAX_DIR_SIZE

def list_json_files(path):
    files = listdir(path)
    json_files = []
    for file in files:
        if file.endswith('.json'):
            json_files.append(file)
    
    return json_files

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def get_dirs(path_):
    files = listdir(path_)
    dirs  = [dir_ for dir_ in files if path.isdir(path.join(path_, dir_))]

    return dirs

def get_random_subset(trials_dict, size):
    if len(trials_dict) <= size:
        return trials_dict
    
    random.seed(constants.RANDOM_SEED)

    return dict(random.sample(trials_dict.items(), size))

def load_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)

def save_to_json(file, data):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)