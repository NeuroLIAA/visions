import json
import random
import constants

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