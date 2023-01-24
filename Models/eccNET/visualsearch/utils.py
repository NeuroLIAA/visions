import json
import pandas as pd

def rescale_coordinate(value, old_size, new_size):
    return int((value / old_size) * new_size)

def load_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)

def load_from_dataset(dataset_path, filename):
    return load_dict_from_json(dataset_path / filename)

def save_scanpaths(scanpaths, output_path, filename='Scanpaths.json'):
    if not output_path.exists(): output_path.mkdir(parents=True)
    
    save_to_json(output_path / filename, scanpaths)

def save_to_json(file, data):
    with file.open('w') as json_file:
        json.dump(data, json_file, indent=4)

def save_probability_map(fix_number, img_name, probability_map, output_path):
    save_path = output_path / 'probability_maps' / img_name[:-4]
    if not save_path.exists(): save_path.mkdir(parents=True)

    probability_map_df = pd.DataFrame(probability_map)
    probability_map_df.to_csv(str(save_path / f'fixation_{fix_number}.csv'), index=False)

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
                 'model_subname': ''
    }

    return model_cfg

def build_trialscanpath(fixations, tg_found, tg_bbox, img_size, max_fix, receptive_size, tg_object, dataset):
    scanpath = {'subject' : 'eccNET Model', 'dataset' : dataset, 'image_height' : img_size[0], 'image_width' : img_size[1], \
            'receptive_height' : receptive_size, 'receptive_width': receptive_size, 'target_found' : tg_found, 'target_bbox' : tg_bbox, \
            'X' : fixations[:, 1], 'Y' : fixations[:, 0], 'target_object' : tg_object, 'max_fixations' : max_fix
    }
    
    return scanpath

def load_human_scanpaths(human_subject, img_size, model_img_size, human_scanpaths_dir):
    if human_subject is None:
        return {}

    human_scanpaths_files = [human_scanpaths_file.name for human_scanpaths_file in human_scanpaths_dir.glob('*.json')]
    human_subject_str     = str(human_subject).zfill(2)
    human_subject_file    = f'subj{human_subject_str}_scanpaths.json'
    if not human_subject_file in human_scanpaths_files:
        raise NameError(f'Scanpaths for subject {human_subject_str} not found!')
    
    human_scanpaths = load_dict_from_json(human_scanpaths_dir / human_subject_file)
    # Convert to int and rescale coordinates
    for trial in human_scanpaths:
        scanpath = human_scanpaths[trial]
        scanpath['X'] = [rescale_coordinate(x_coord, img_size[1], model_img_size[1]) for x_coord in scanpath['X']]
        scanpath['Y'] = [rescale_coordinate(y_coord, img_size[0], model_img_size[0]) for y_coord in scanpath['Y']]

    return human_scanpaths

def keep_human_trials(human_scanpaths, trials_properties):
    human_trials_properties = []
    for trial in trials_properties:
        if trial['image'] in human_scanpaths:
            human_trials_properties.append(trial)
    
    if not human_trials_properties:
        raise ValueError('Human subject does not have any scanpaths for the images specified')

    return human_trials_properties

def get_scanpath(human_scanpaths, img_name):
    scanpath = {}
    if human_scanpaths:
        scanpath = human_scanpaths[img_name]
    
    return scanpath