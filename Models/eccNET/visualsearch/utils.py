from pathlib import Path
import json

def rescale_coordinate(value, old_size, new_size):
    return int((value / old_size) * new_size)

def load_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)

def load_from_dataset(dataset_path, filename):
    return load_dict_from_json(dataset_path / filename)

def build_expinfo(num_images, max_fixations, img_size, target_size, eye_res, deg_to_pixel, dog_size, weight_pattern='l'):
    exp_info = {'eye_res': eye_res,
                'stim_shape': (img_size[0], img_size[1], 3),
                'tar_shape': (target_size[0], target_size[1], 3),
                'ior_size': 2 * target_size[0],
                'NumStimuli': num_images,
                'NumFix': max_fixations,
                'gt_mask': None,
                'fix': None,
                'bg_value': 0,
                'rev_img_flag': 0,
                'deg2px': deg_to_pixel,
                'weight_pattern': weight_pattern,
                'dog_size': dog_size,
                's_ratio': None
    }

    return exp_info

def build_modelcfg(ecc_param, vgg16_weights):
    model_cfg = {'eccParam': ecc_param,
                 'ecc_depth': len(ecc_param['ecc_slope']),
                 'out_layer': [1, 1, 1],
                 'comp_layer': 'diff',
                 'vgg_model_path': vgg16_weights,
                 'model_subname': ""
    }

    return model_cfg