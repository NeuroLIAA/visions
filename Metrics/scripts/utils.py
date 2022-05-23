import json
import random
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, scandir, makedirs
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
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
    fig.set_size_inches(12, 5)
    plt.savefig(path.join(save_path, filename))
    plt.show()

def average_results(datasets_results_dict, save_path, filename):
    final_table = {}
    metrics = ['AUCperf', 'AvgMM', 'AUChsp', 'NSShsp', 'IGhsp', 'LLhsp']
    number_of_datasets = len(datasets_results_dict)
    for dataset in datasets_results_dict:
        dataset_res   = datasets_results_dict[dataset]
        human_aucperf = dataset_res['Humans']['AUCperf']

        for model in dataset_res:
            if model == 'Humans': continue
            if not model in final_table:
                final_table[model] = {}
                for metric in metrics:
                    final_table[model][metric] = 0
                final_table[model]['Score'] = 0
            
            # AUCperf is expressed as 1 subtracted the absolute difference between Human and model's AUCperf, maximizing the score of those models who were closest to human subjects
            dif_aucperf = 1 - abs(human_aucperf - dataset_res[model]['AUCperf'])

            final_table[model]['AUCperf'] += dif_aucperf / number_of_datasets
            for metric in metrics[1:]:
                final_table[model][metric] += dataset_res[model][metric] / number_of_datasets
    
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

def create_dirs(filepath):
    dir_ = path.dirname(filepath)
    if len(dir_) > 0 and not path.exists(dir_):
        makedirs(dir_)  

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

def load_pickle(pickle_filepath):
    with open(pickle_filepath, 'rb') as fp:
        return pickle.load(fp)  

def load_dict_from_json(json_file_path):
    if not path.exists(json_file_path):
        return {}
    else:
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)

def save_to_pickle(data, filepath):
    create_dirs(filepath)

    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)

def save_to_json(filepath, data):
    create_dirs(filepath)

    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def save_to_csv(data, filepath):
    create_dirs(filepath)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def get_dims(model_trial, subject_trial, key):
    " Lower bound for receptive size "
    if key == 'receptive' and model_trial[key + '_width'] > subject_trial[key + '_width']:
        return (subject_trial[key + '_height'], subject_trial[key + '_width'])
    
    return (model_trial[key + '_height'], model_trial[key + '_width'])

def get_scanpath_time(trial_info, length):
    if 'T' in trial_info:
        scanpath_time = [t * 0.0001 for t in trial_info['T']]
    else:
        # Dummy
        scanpath_time = [0.3] * length
    
    return scanpath_time

def rescale_and_crop(trial_info, new_size, receptive_size):
    trial_scanpath_X = [rescale_coordinate(x, trial_info['image_width'], new_size[1]) for x in trial_info['X']]
    trial_scanpath_Y = [rescale_coordinate(y, trial_info['image_height'], new_size[0]) for y in trial_info['Y']]

    image_size       = (trial_info['image_height'], trial_info['image_width'])
    target_bbox      = trial_info['target_bbox']
    target_bbox      = [rescale_coordinate(target_bbox[i], image_size[i % 2 == 1], new_size[i % 2 == 1]) for i in range(len(target_bbox))]

    trial_scanpath_X, trial_scanpath_Y = collapse_fixations(trial_scanpath_X, trial_scanpath_Y, receptive_size)
    trial_scanpath_X, trial_scanpath_Y = crop_scanpath(trial_scanpath_X, trial_scanpath_Y, target_bbox, receptive_size)

    return trial_scanpath_X, trial_scanpath_Y        

def rescale_coordinate(value, old_size, new_size):
    return int((value / old_size) * new_size)

def between_bounds(target_bbox, fix_y, fix_x, receptive_size):
    return target_bbox[0] <= fix_y + receptive_size[0] // 2 and target_bbox[2] >= fix_y - receptive_size[0] // 2 and \
        target_bbox[1] <= fix_x + receptive_size[1] // 2 and target_bbox[3] >= fix_x - receptive_size[1] // 2

def crop_scanpath(scanpath_x, scanpath_y, target_bbox, receptive_size):
    index = 0
    for fixation in zip(scanpath_y, scanpath_x):
        if between_bounds(target_bbox, fixation[0], fixation[1], receptive_size):
            break
        index += 1
    
    cropped_scanpath_x = list(scanpath_x[:index + 1])
    cropped_scanpath_y = list(scanpath_y[:index + 1])
    return cropped_scanpath_x, cropped_scanpath_y

def collapse_fixations(scanpath_x, scanpath_y, receptive_size):
    collapsed_scanpath_x = list(scanpath_x)
    collapsed_scanpath_y = list(scanpath_y)
    index = 0
    while index < len(collapsed_scanpath_x) - 1:
        abs_difference_x = [abs(fix_1 - fix_2) for fix_1, fix_2 in zip(collapsed_scanpath_x, collapsed_scanpath_x[1:])]
        abs_difference_y = [abs(fix_1 - fix_2) for fix_1, fix_2 in zip(collapsed_scanpath_y, collapsed_scanpath_y[1:])]

        if abs_difference_x[index] < receptive_size[1] / 2 and abs_difference_y[index] < receptive_size[0] / 2:
            new_fix_x = (collapsed_scanpath_x[index] + collapsed_scanpath_x[index + 1]) / 2
            new_fix_y = (collapsed_scanpath_y[index] + collapsed_scanpath_y[index + 1]) / 2
            collapsed_scanpath_x[index] = new_fix_x
            collapsed_scanpath_y[index] = new_fix_y
            del collapsed_scanpath_x[index + 1]
            del collapsed_scanpath_y[index + 1]
        else:
            index += 1

    return collapsed_scanpath_x, collapsed_scanpath_y

def aggregate_scanpaths(subjects_scanpaths_path, image_name, excluded_subject='None'):
    subjects_scanpaths_files = sorted_alphanumeric(listdir(subjects_scanpaths_path))

    scanpaths_X = []
    scanpaths_Y = []
    for subject_file in subjects_scanpaths_files:
        if excluded_subject in subject_file:
            continue
        subject_scanpaths = load_dict_from_json(path.join(subjects_scanpaths_path, subject_file))
        if image_name in subject_scanpaths:
            trial = subject_scanpaths[image_name]
            scanpaths_X += trial['X']
            scanpaths_Y += trial['Y']

    scanpaths_X = np.array(scanpaths_X)
    scanpaths_Y = np.array(scanpaths_Y)

    return scanpaths_X, scanpaths_Y

def search_bandwidth(values, shape, splits=5):
    """ Perform a grid search to look for the optimal bandwidth (i.e. the one that maximizes log-likelihood) """
    # Define search space (values estimated from previous executions)
    if np.log(shape[0] * shape[1]) < 10:
        bandwidths = 10 ** np.linspace(-1, 1, 100)
    else:
        bandwidths = np.linspace(15, 70, 200)
    
    n_splits = min(values.shape[0], splits)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths}, n_jobs=-1, cv=n_splits)
    grid.fit(values)

    return grid.best_params_['bandwidth']

def load_center_bias_fixations(model_size):
    center_bias_fixs = load_dict_from_json(constants.CENTER_BIAS_FIXATIONS)

    scanpaths_X = np.array([rescale_coordinate(x, constants.CENTER_BIAS_SIZE[1], model_size[1]) for x in center_bias_fixs['X']])
    scanpaths_Y = np.array([rescale_coordinate(y, constants.CENTER_BIAS_SIZE[0], model_size[0]) for y in center_bias_fixs['Y']])

    return scanpaths_X, scanpaths_Y

def gaussian_kde(scanpaths_X, scanpaths_Y, shape, bandwidth=None):
    values = np.vstack([scanpaths_Y, scanpaths_X]).T

    if bandwidth is None:
        bandwidth = search_bandwidth(values, shape)

    X, Y = np.mgrid[0:shape[0], 0:shape[1]] + 0.5
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    gkde   = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)
    scores = np.exp(gkde.score_samples(positions))

    return scores.reshape(shape)