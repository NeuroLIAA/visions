import json
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir, path, scandir
from .. import constants

def plot_table(df, title):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()
    fig.suptitle(title)
    plt.show()

def create_df(dict_):
    return pd.DataFrame.from_dict(dict_)

def dir_is_too_heavy(path):
    nmbytes = sum(d.stat().st_size for d in scandir(path) if d.is_file()) / 2**20
    
    return nmbytes > constants.MAX_DIR_SIZE

def is_contained_in(json_file_1, json_file_2):
    if not (path.exists(json_file_1) and path.exists(json_file_2)):
        return False
    
    dict_1 = load_dict_from_json(json_file_1)
    dict_2 = load_dict_from_json(json_file_2)

    return all(image_name in list(dict_2.keys()) for image_name in list(dict_1.keys()))

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

def update_dict(dic, key, data):
    if key in dic:
        dic[key].update(data)
    else:
        dic[key] = data

def load_dict_from_json(json_file_path):
    if not path.exists(json_file_path):
        return {}
    else:
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)

def save_to_json(file, data):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)