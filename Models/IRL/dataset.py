import numpy as np
import constants
import json
from os import path, listdir
from irl_dcb import utils
from irl_dcb.data import LHF_IRL
from irl_dcb.build_belief_maps import build_belief_maps

def process_trials(trials_properties, images_dir, human_scanpaths, new_image_size, grid_size, DCB_dir_HR, DCB_dir_LR):
    bbox_annos = {}
    iteration  = 1
    for image_data in list(trials_properties):
        # If the target isn't categorized, remove it
        if image_data['target_object'] == 'TBD':
            trials_properties.remove(image_data)
            continue
        
        # If the model is going to follow a given subject's scanpaths, only keep those scanpaths related to the subject
        if human_scanpaths and not (image_data['image'] in human_scanpaths):
            trials_properties.remove(image_data)
            continue
            
        old_image_size = (image_data['image_height'], image_data['image_width'])

        # Rescale everything to image size used
        rescale_coordinates(image_data, old_image_size, new_image_size)
        
        # Add trial's bounding box info to dict
        category_and_image_name             = image_data['target_object'] + '_' + image_data['image']
        bbox_annos[category_and_image_name] = (image_data['target_matched_column'], image_data['target_matched_row'], image_data['target_width'], image_data['target_height'])

        # Create belief maps for image if necessary
        check_and_build_belief_maps(image_data['image'], images_dir, DCB_dir_HR, DCB_dir_LR, new_image_size, grid_size, iteration, total=len(trials_properties))
        
        iteration += 1
    
    return bbox_annos

def check_and_build_belief_maps(image_name, images_dir, DCB_dir_HR, DCB_dir_LR, new_image_size, grid_size, iter_number, total):
    img_belief_maps_file = image_name[:-4] + '.pth.tar'
    high_res_belief_maps = path.join(DCB_dir_HR, img_belief_maps_file)
    low_res_belief_maps  = path.join(DCB_dir_LR, img_belief_maps_file)
    if not (path.exists(high_res_belief_maps) and path.exists(low_res_belief_maps)):
        print('Building belief maps for image ' + image_name + ' (' + str(iter_number) + '/' + str(total) + ')')
        build_belief_maps(image_name, images_dir, (new_image_size[1], new_image_size[0]), grid_size, constants.SIGMA_BLUR, constants.NUMBER_OF_BELIEF_MAPS, DCB_dir_HR, DCB_dir_LR)    

def rescale_coordinates(image_data, old_image_size, new_image_size):
    old_image_height = old_image_size[0]
    old_image_width  = old_image_size[1]
    new_image_height = new_image_size[0]
    new_image_width  = new_image_size[1]

    image_data['target_matched_column']   = utils.rescale_coordinate(image_data['target_matched_column'], old_image_width, new_image_width)
    image_data['target_matched_row']      = utils.rescale_coordinate(image_data['target_matched_row'], old_image_height, new_image_height)
    image_data['target_width']            = utils.rescale_coordinate(image_data['target_width'], old_image_width, new_image_width)
    image_data['target_height']           = utils.rescale_coordinate(image_data['target_height'], old_image_height, new_image_height)
    image_data['initial_fixation_column'] = utils.rescale_coordinate(image_data['initial_fixation_column'], old_image_width, new_image_width)
    image_data['initial_fixation_row']    = utils.rescale_coordinate(image_data['initial_fixation_row'], old_image_height, new_image_height)

    # Save new image size
    image_data['image_width']  = new_image_width
    image_data['image_height'] = new_image_height

def process_eval_data(trials_properties, DCB_HR_dir, DCB_LR_dir, target_annos, hparams):
    target_init_fixs = {}
    for image_data in trials_properties:
        key = image_data['target_object'] + '_' + image_data['image']
        target_init_fixs[key] = (image_data['initial_fixation_column'] / image_data['image_width'],
                                image_data['initial_fixation_row'] / image_data['image_height'])

    # Since the model was trained for these specific categories, the list must always be the same, regardless of the dataset
    target_objects = ['bottle', 'bowl', 'car', 'chair', 'clock', 'cup', 'fork', 'keyboard', 'knife', 'laptop', \
        'microwave', 'mouse', 'oven', 'potted plant', 'sink', 'stop sign', 'toilet', 'tv']
    # target_objects = list(np.unique([x['target_object'] for x in trials_properties]))

    catIds = dict(zip(target_objects, list(range(len(target_objects)))))

    test_task_img_pair = np.unique([traj['target_object'] + '-' + traj['image'] for traj in trials_properties])

    # Load image data
    test_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                   test_task_img_pair, target_annos,
                                   hparams.Data, catIds)
    return {
            'catIds': catIds,
            'img_test': test_img_dataset,
        }

def load_human_scanpaths(human_scanpaths_dir, human_subject, grid_size, patch_size):
    if human_subject is None:
        return {}

    human_scanpaths_files = listdir(human_scanpaths_dir)
    human_subject_str     = str(human_subject)
    if human_subject < 10: human_subject_str = '0' + human_subject_str
    human_subject_file    = 'subj' + human_subject_str + '_scanpaths.json'
    if not human_subject_file in human_scanpaths_files:
        raise NameError('Scanpaths for human subject ' + human_subject_str + ' not found!')
    
    human_scanpaths = utils.load_dict_from_json(path.join(human_scanpaths_dir, human_subject_file))

    rescale_scanpaths(human_scanpaths, grid_size, patch_size)

    return human_scanpaths    

def rescale_scanpaths(human_scanpaths, grid_size, patch_size):
    for key in human_scanpaths.keys():        
        scanpath = human_scanpaths[key]
        scanpath['X'], scanpath['Y'] = [list(coords) for coords in zip(*list(map(utils.map_to_cell, zip(scanpath['X'], scanpath['Y']))))]
        # Convert to int so it can be saved in JSON format
        scanpath['X'] = [int(x_coord) for x_coord in scanpath['X']]
        scanpath['Y'] = [int(y_coord) for y_coord in scanpath['Y']]