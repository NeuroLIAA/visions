import argparse
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os import path, listdir

datasets_path = '../Datasets'
results_path  = '../Results'

def plot_probability_maps(human_scanpath, probability_maps_path, model_name, title):
    probability_maps = sorted_alphanumeric(listdir(probability_maps_path))
    image_size  = [human_scanpath['image_height'], human_scanpath['image_width']]
    target_bbox = human_scanpath['target_bbox']

    fig, ax = plt.subplots(nrows=1, ncols=len(probability_maps), figsize=[25,5])
    for index, prob_map in enumerate(probability_maps):
        prob_map_df = pd.read_csv(path.join(probability_maps_path, prob_map))
        grid_size   = prob_map_df.shape
        scanpath_x, scanpath_y = rescale_scanpath(human_scanpath, grid_size)

        rescaled_bbox = [rescale_coordinate(target_bbox[i], image_size[i % 2 == 1], grid_size[i % 2 == 1]) for i in range(len(target_bbox))]
        target_height = rescaled_bbox[2] - rescaled_bbox[0]
        target_width  = rescaled_bbox[3] - rescaled_bbox[1]
        rescaled_bbox = [rescaled_bbox[1], rescaled_bbox[0], target_width, target_height]

        if model_name == 'IVSN':
            fixation_size = human_scanpath['receptive_width'] // 2
        else:
            fixation_size = 1

        initial_color  = 'red'
        scanpath_color = 'darkorange'

        ax[index].imshow(prob_map_df)
        for i in range(index + 2):
            if i > 0:
                ax[index].arrow(scanpath_x[i - 1], scanpath_y[i - 1], scanpath_x[i] - scanpath_x[i - 1], scanpath_y[i] - scanpath_y[i - 1], width=0.1, color=scanpath_color, alpha=0.5)

        for i in range(index + 2):
            if i == 0:
                edge_color = initial_color
            else:
                edge_color = scanpath_color
            circle = plt.Circle((scanpath_x[i], scanpath_y[i]),
                                radius=fixation_size,
                                edgecolor=edge_color,
                                facecolor='none',
                                alpha=0.5)
            ax[index].add_patch(circle)

        # Draw target's bbox
        rect = Rectangle((rescaled_bbox[0], rescaled_bbox[1]), rescaled_bbox[2], rescaled_bbox[3], alpha=0.7, edgecolor='red', facecolor='none', linewidth=2)
        ax[index].add_patch(rect)

        ax[index].axis('off')

    fig.suptitle(title)
    plt.show()
    plt.close()

def rescale_coordinate(value, old_size, new_size):
    return (value / old_size) * new_size

def rescale_scanpath(scanpath, image_size):
    scanpath_x_rescaled = [rescale_coordinate(x, scanpath['image_width'], image_size[1]) for x in scanpath['X']]
    scanpath_y_rescaled = [rescale_coordinate(y, scanpath['image_height'], image_size[0]) for y in scanpath['Y']]
    scanpath_x_rescaled = [int(x_coord) for x_coord in scanpath_x_rescaled]
    scanpath_y_rescaled = [int(y_coord) for y_coord in scanpath_y_rescaled]

    return scanpath_x_rescaled, scanpath_y_rescaled

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def get_subject_str(subject_id):
    subject_str = str(subject_id)
    if subject_id < 10:
        subject_str = '0' + subject_str

    return subject_str

def load_human_scanpath(subject_id, dataset_name, image_name):
    subject_str = get_subject_str(subject_id)
    subject_scanpaths_file = path.join(path.join(path.join(datasets_path, dataset_name), 'human_scanpaths'), 'subj' + subject_str + '_scanpaths.json')
    subject_scanpaths = load_dict_from_json(subject_scanpaths_file)

    if not image_name in subject_scanpaths:
        raise ValueError(image_name + ' not found in subject\'s scanpaths')
    
    return subject_scanpaths[image_name]

def load_prob_maps_path(subject_id, dataset_name, model_name, image_name):
    subject_str     = get_subject_str(subject_id)
    probability_maps_path = path.join(results_path, dataset_name + '_dataset', model_name, 'subjects_predictions', \
         'subject_' + subject_str, 'probability_maps', image_name[:-4])
    
    if not path.exists(probability_maps_path):
        raise ValueError('Path not found for subject ' + subject_str + ' in ' + args.model + ' on ' + args.dataset + ' dataset')
    
    return probability_maps_path

def load_dict_from_json(json_file_path):
    if not path.exists(json_file_path):
        return {}
    else:
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='Name of the visual search model')
    parser.add_argument('-subject', type=int, help='On which human subject plot the probability maps')
    parser.add_argument('-dataset', type=str, help='Name of the dataset')
    parser.add_argument('-img', type=str, help='Name of the image on which to draw the scanpath (write \'notfound\' to plot target not found images')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    human_scanpath  = load_human_scanpath(args.subject, args.dataset, args.img)
    prob_maps_path  = load_prob_maps_path(args.subject, args.dataset, args.model, args.img)

    title = args.img + '; subject: ' + str(args.subject) + ', model: ' + args.model + ', dataset: ' + args.dataset

    plot_probability_maps(human_scanpath, prob_maps_path, args.model, title)