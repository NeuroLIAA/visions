import json
import numpy  as np
import pandas as pd
from os import makedirs, path, remove, listdir
from skimage import io, transform, img_as_ubyte, color


def is_coloured(image):
    return len(image.shape) > 2

def to_grayscale(image):
    return color.rgb2gray(image)

def load_data_from_checkpoint(output_path):
    checkpoint_file = path.join(output_path, 'checkpoint.json')
    scanpaths     = {}
    targets_found = 0
    time_elapsed  = 0
    if path.exists(checkpoint_file):
        checkpoint    = load_from_json(checkpoint_file)
        scanpaths     = checkpoint['scanpaths']
        targets_found = checkpoint['targets_found']
        time_elapsed  = checkpoint['time_elapsed']
    
    return scanpaths, targets_found, time_elapsed

def save_checkpoint(config, scanpaths, targets_found, trials_properties, time_elapsed, output_path):
    checkpoint = {}
    checkpoint['configuration']     = config
    checkpoint['targets_found']     = targets_found
    checkpoint['time_elapsed']      = time_elapsed
    checkpoint['scanpaths']         = scanpaths
    checkpoint['trials_properties'] = remove_trials_already_processed(trials_properties, scanpaths)

    save_to_json(path.join(output_path, 'checkpoint.json'), checkpoint)
    print('\nCheckpoint saved at ' + output_path)
    print('Run the script again to resume execution')

def erase_checkpoint(output_path):
    checkpoint_file = path.join(output_path, 'checkpoint.json')
    if path.exists(checkpoint_file):
        remove(checkpoint_file)

def remove_trials_already_processed(trials_properties, scanpaths):
    remaining_trials = []
    for trial in trials_properties:
        image_name = trial['image']
        if not image_name in scanpaths:
            remaining_trials.append(trial)
    
    return remaining_trials

def save_scanpaths(output_path, scanpaths, filename='Scanpaths.json'):
    save_to_json(path.join(output_path, filename), scanpaths)

def save_to_json(file, data):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)

def load_image(img_path, name, image_size='default'):
    img = io.imread(path.join(img_path, name))

    if image_size != 'default':
        img = img_as_ubyte(transform.resize(img, image_size))

    return img

def save_probability_map(output_path, image_name, probability_map, fixation_number):
    save_path = path.join(output_path, 'probability_maps', image_name[:-4])
    if not path.exists(save_path):
        makedirs(save_path)

    posterior_df = pd.DataFrame(probability_map)
    posterior_df.to_csv(path.join(save_path, 'fixation_' + str(fixation_number + 1) + '.csv'), index=False)

def exists_probability_maps_for_image(image_name, output_path):
    probability_maps_path = path.join(output_path, path.join('probability_maps', image_name[:-4]))
    probability_maps = []
    if path.exists(probability_maps_path):
        probability_maps = listdir(probability_maps_path)

    return bool(probability_maps)

def save_similarity_map(output_path, filename, target_similarity_map):
    if not path.exists(output_path):
        makedirs(output_path)

    io.imsave(path.join(output_path, filename), img_as_ubyte(target_similarity_map), check_contrast=False)

def add_white_gaussian_noise(image, snr_db):
    """ Input:
            image (2D array) : image where noise will be added
            snr_db (int)     : signal-to-noise ratio, in dB
        Output: 
            image (2D array) : input image with noise added
    """
    snr_watts = 10 ** (snr_db / 10)
    image_size = (image.shape[0], image.shape[1])

    noisy_image = image + np.normal(0, np.sqrt(snr_watts), shape=image_size)

    return noisy_image

def add_scanpath_to_dict(image_name, image_scanpath, target_bbox, target_object, grid, config, dataset_name, dict_):
    target_found = image_scanpath['target_found']
    scanpath_x   = image_scanpath['scanpath_x']
    scanpath_y   = image_scanpath['scanpath_y']
    target_bbox_in_grid = np.empty(len(target_bbox), dtype=np.int)
    target_bbox_in_grid[0], target_bbox_in_grid[1] = grid.map_to_cell((target_bbox[0], target_bbox[1]))
    target_bbox_in_grid[2], target_bbox_in_grid[3] = grid.map_to_cell((target_bbox[2], target_bbox[3]))

    dict_[image_name] = {'subject' : 'cIBS Model', 'dataset' : dataset_name, 'image_height' : int(grid.size()[0]), 'image_width' : int(grid.size()[1]), \
        'receptive_height' : 1, 'receptive_width' : 1, 'target_found' : target_found, 'target_bbox' : target_bbox_in_grid.tolist(), \
                 'X' : list(map(int, scanpath_x)), 'Y' : list(map(int, scanpath_y)), 'target_object' : target_object, 'max_fixations' : config['max_saccades'] + 1
        }

def are_within_boundaries(top_left_coordinates, bottom_right_coordinates, top_left_coordinates_to_compare, bottom_right_coordinates_to_compare):
    return top_left_coordinates[0] >= top_left_coordinates_to_compare[0] and top_left_coordinates[1] >= top_left_coordinates_to_compare[1] \
         and bottom_right_coordinates[0] < bottom_right_coordinates_to_compare[0] and bottom_right_coordinates[1] < bottom_right_coordinates_to_compare[1]

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

def rescale_scanpaths(grid, human_scanpaths):
    for trial in human_scanpaths:        
        scanpath = human_scanpaths[trial]
        scanpath['X'], scanpath['Y'] = rescale_and_crop(scanpath, grid.size(), [1, 1])
        # Convert to int so it can be saved in JSON format
        scanpath['X'] = [int(x_coord) for x_coord in scanpath['X']]
        scanpath['Y'] = [int(y_coord) for y_coord in scanpath['Y']]