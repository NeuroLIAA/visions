import json
import torch
import numpy as np
from math import floor
from os import path, makedirs

def add_scanpath_to_dict(model_name, image_name, image_size, scanpath_x, scanpath_y, target_object, cell_size, max_saccades, dataset_name, dict_):

    dict_[image_name] = {'subject' : model_name, 'dataset' : dataset_name, 'image_height' : image_size[0], 'image_width' : image_size[1], \
        'receptive_height' : cell_size, 'receptive_width' : cell_size, 'target_found' : False, 'target_bbox' : np.zeros(shape=4), \
                 'X' : list(map(int, scanpath_x)), 'Y' : list(map(int, scanpath_y)), 'target_object' : target_object, 'max_fixations' : max_saccades + 1
        }

def save_scanpaths(output_path, scanpaths):
    if not path.exists(output_path):
        makedirs(output_path)
    save_to_json(output_path + 'Scanpaths.json', scanpaths)

def save_to_json(file, data):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def actions2scanpaths(actions, patch_num, im_w, im_h, dataset_name, patch_size, max_saccades):
    scanpaths = {}
    for traj in actions:
        task_name, img_name, condition, actions = traj
        actions = actions.to(dtype=torch.float32)
        py = (actions // patch_num[0]) / float(patch_num[1])
        px = (actions % patch_num[0]) / float(patch_num[0])
        fixs = torch.stack([px, py])
        fixs = np.concatenate([np.array([[0.5], [0.5]]),
                               fixs.cpu().numpy()],
                              axis=1)
        add_scanpath_to_dict('IRL Model', img_name, (im_h,im_w), fixs[0] * im_w, fixs[1] * im_h, task_name, patch_size, max_saccades, dataset_name, scanpaths)
        
    
    return scanpaths

def cutFixOnTarget(trajs, target_annos):

    for image_name in trajs:
        traj = trajs[image_name]
        key = traj['target_object'] + '_' + image_name
        bbox = target_annos[key]
        traj_len = get_num_step2target(traj['X'], traj['Y'], bbox)
        if traj_len != 1000:
            traj['target_found'] = True
        traj['X'] = traj['X'][:traj_len]
        traj['Y'] = traj['Y'][:traj_len]
        traj['target_bbox'] = [bbox[1],bbox[0],bbox[1] + bbox[3], bbox[0] + bbox[2]]

def get_num_step2target(X, Y, bbox):
    X, Y = np.array(X), np.array(Y)
    on_target_X = np.logical_and(X > bbox[0], X < bbox[0] + bbox[2])
    on_target_Y = np.logical_and(Y > bbox[1], Y < bbox[1] + bbox[3])
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    else:
        return 1000  # some big enough number

def rescale_coordinate(value, old_size, new_size):
    return floor((value / old_size) * new_size)

    