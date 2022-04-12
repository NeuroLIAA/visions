import numpy as np
from . import constants
from os import path, listdir
from .irl_dcb import utils
from .irl_dcb.data import LHF_IRL
from .irl_dcb.build_belief_maps import build_belief_maps

def process_trials(trials_properties, images_dir, human_scanpaths, new_image_size, grid_size, DCB_dir_HR, DCB_dir_LR):
    bbox_annos = {}
    iteration  = 1
    for trial in list(trials_properties):
        # If the target isn't categorized, remove it
        if trial['target_object'] == 'TBD':
            trials_properties.remove(trial)
            continue
        
        # If the model is going to follow a given subject's scanpaths, only keep those scanpaths related to the subject
        if human_scanpaths and not (trial['image'] in human_scanpaths):
            trials_properties.remove(trial)
            continue
            
        old_image_size = (trial['image_height'], trial['image_width'])

        # Rescale everything to image size used
        rescale_coordinates(trial, old_image_size, new_image_size)
        
        # Add trial's bounding box info to dict
        category_and_image_name             = trial['target_object'] + '_' + trial['image']
        bbox_annos[category_and_image_name] = (trial['target_matched_column'], trial['target_matched_row'], trial['target_width'], trial['target_height'])

        # Create belief maps for image if necessary
        check_and_build_belief_maps(trial['image'], images_dir, DCB_dir_HR, DCB_dir_LR, new_image_size, grid_size, iteration, total=len(trials_properties))
        
        iteration += 1
    
    return bbox_annos

def check_and_build_belief_maps(image_name, images_dir, DCB_dir_HR, DCB_dir_LR, new_image_size, grid_size, iter_number, total):
    img_belief_maps_file = image_name[:-4] + '.pth.tar'
    high_res_belief_maps = path.join(DCB_dir_HR, img_belief_maps_file)
    low_res_belief_maps  = path.join(DCB_dir_LR, img_belief_maps_file)
    if not (path.exists(high_res_belief_maps) and path.exists(low_res_belief_maps)):
        print('Building belief maps for image ' + image_name + ' (' + str(iter_number) + '/' + str(total) + ')')
        build_belief_maps(image_name, images_dir, (new_image_size[1], new_image_size[0]), grid_size, constants.SIGMA_BLUR, constants.NUMBER_OF_BELIEF_MAPS, DCB_dir_HR, DCB_dir_LR)    

def rescale_coordinates(trial, old_image_size, new_image_size):
    old_image_height = old_image_size[0]
    old_image_width  = old_image_size[1]
    new_image_height = new_image_size[0]
    new_image_width  = new_image_size[1]

    trial['target_matched_column']   = utils.rescale_coordinate(trial['target_matched_column'], old_image_width, new_image_width)
    trial['target_matched_row']      = utils.rescale_coordinate(trial['target_matched_row'], old_image_height, new_image_height)
    trial['target_width']            = utils.rescale_coordinate(trial['target_width'], old_image_width, new_image_width)
    trial['target_height']           = utils.rescale_coordinate(trial['target_height'], old_image_height, new_image_height)
    trial['initial_fixation_column'] = utils.rescale_coordinate(trial['initial_fixation_column'], old_image_width, new_image_width)
    trial['initial_fixation_row']    = utils.rescale_coordinate(trial['initial_fixation_row'], old_image_height, new_image_height)

    # Save new image size
    trial['image_width']  = new_image_width
    trial['image_height'] = new_image_height

def process_eval_data(trials_properties, human_scanpaths, DCB_HR_dir, DCB_LR_dir, target_annos, grid_size, hparams):
    target_init_fixs = {}
    for trial in trials_properties:
        key = trial['target_object'] + '_' + trial['image']
        if human_scanpaths:
            initial_fix = (human_scanpaths[trial['image']]['X'][0] / grid_size[1], human_scanpaths[trial['image']]['Y'][0] / grid_size[0])
        else:
            initial_fix = (trial['initial_fixation_column'] / trial['image_width'], trial['initial_fixation_row'] / trial['image_height'])
        target_init_fixs[key] = initial_fix

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

def load_human_scanpaths(human_scanpaths_dir, human_subject, grid_size):
    if human_subject is None:
        return {}

    human_scanpaths_files = listdir(human_scanpaths_dir)
    human_subject_str     = str(human_subject)
    if human_subject < 10: human_subject_str = '0' + human_subject_str
    human_subject_file    = 'subj' + human_subject_str + '_scanpaths.json'
    if not human_subject_file in human_scanpaths_files:
        raise NameError('Scanpaths for human subject ' + human_subject_str + ' not found!')
    
    human_scanpaths = utils.load_dict_from_json(path.join(human_scanpaths_dir, human_subject_file))

    rescale_scanpaths(human_scanpaths, grid_size)

    return human_scanpaths    

def rescale_scanpaths(human_scanpaths, grid_size):
    for key in human_scanpaths:        
        scanpath     = human_scanpaths[key]
        image_height = scanpath['image_height']
        image_width  = scanpath['image_width']
        image_size   = (image_height, image_width)
        scanpath['X'] = [int(utils.rescale_coordinate(x_coord, image_width, grid_size[1])) for x_coord in scanpath['X']]
        scanpath['Y'] = [int(utils.rescale_coordinate(y_coord, image_height, grid_size[0])) for y_coord in scanpath['Y']]
        scanpath['target_bbox'] = [int(utils.rescale_coordinate(scanpath['target_bbox'][i], image_size[i % 2 == 1], grid_size[i % 2 == 1])) for i in range(len(scanpath['target_bbox']))]