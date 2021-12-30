import json
import random
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, scandir
from .. import constants

def plot_table(df, title, save_path, filename):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, colColours=['peachpuff'] * len(df.columns), rowColours=['peachpuff'] * len(df.index), loc='center')
    table.auto_set_column_width(list(range(len(df.columns))))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    fig.tight_layout()
    fig.suptitle(title)
    fig.set_size_inches(10.5, 5)
    plt.savefig(path.join(save_path, filename))
    plt.show()

def average_results(datasets_results_dict, save_path, filename):
    final_table = {}
    number_of_datasets = len(datasets_results_dict)
    for dataset in datasets_results_dict:
        dataset_res   = datasets_results_dict[dataset]
        human_aucperf = dataset_res['Humans']['AUCperf']

        for model in dataset_res:
            if model == 'Humans': continue
            if not model in final_table:
                final_table[model] = {'AUCperf': 0, 'AvgMM': 0, 'AUChsp': 0, 'NSShsp': 0, 'Score': 0}
            
            # AUCperf is expressed as 1 subtracted the absolute difference between Human and model's AUCperf, maximizing the score of those models who were closest to human subjects
            dif_aucperf = 1 - abs(human_aucperf - dataset_res[model]['AUCperf'])

            final_table[model]['AUCperf'] += dif_aucperf / number_of_datasets
            final_table[model]['AvgMM']   += dataset_res[model]['AvgMM'] / number_of_datasets
            final_table[model]['AUChsp']  += dataset_res[model]['AUChsp'] / number_of_datasets
            final_table[model]['NSShsp']  += dataset_res[model]['NSShsp'] / number_of_datasets
    
    # Average and round values
    for model in final_table:
        final_score = 0
        number_of_metrics = 0

        scores = final_table[model]
        for metric in scores:
            if scores[metric] != 0:
                final_score       += scores[metric]
                number_of_metrics += 1
            scores[metric] = np.round(scores[metric], 3)
        final_table[model]['Score'] = np.round(final_score / number_of_metrics, 3)

    save_to_json(path.join(save_path, filename), final_table)
    final_table = create_df(final_table).T

    return final_table.sort_values(by=['Score'], ascending=False)

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