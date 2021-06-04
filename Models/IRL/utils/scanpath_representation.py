import json
from os import path, makedirs

def add_scanpath_to_dict(model_name, image_name, image_size, image_scanpath, target_bbox, cell_size, max_saccades, dataset_name, dict_):
    target_found = image_scanpath['target_found']
    scanpath_x   = image_scanpath['scanpath_x']
    scanpath_y   = image_scanpath['scanpath_y']

    dict_[image_name] = {'subject' : model_name, 'dataset' : dataset_name, 'image_height' : image_size[0], 'image_width' : image_size[1], \
        'receptive_height' : cell_size, 'receptive_width' : cell_size, 'target_found' : target_found, 'target_bbox' : target_bbox, \
                 'X' : list(map(int, scanpath_x)), 'Y' : list(map(int, scanpath_y)), 'target_object' : 'TBD', 'max_fixations' : max_saccades + 1
        }

def save_scanpaths(output_path, scanpaths):
    if not path.exists(output_path):
        makedirs(output_path)
    save_to_json(output_path + 'Scanpaths.json', scanpaths)

def save_to_json(file, data):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
