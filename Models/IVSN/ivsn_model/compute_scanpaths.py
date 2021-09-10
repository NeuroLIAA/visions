import json
import numpy as np
from . import utils
from os import listdir, makedirs, path
from skimage import io, transform, exposure

"""
Puts together data produced by the CNN and creates an attention map for the image, which is used to compute the scanpaths, with a winner-takes-all strategy.
Scanpaths are saved in a JSON file.
"""

def parse_model_data(preprocessed_images_dir, trials_properties, human_scanpaths, image_size, max_fixations, receptive_size, dataset_name, output_path):
    scanpaths = {}
    targets_found = 0
    for trial in trials_properties:
        image_name = trial['image']
        img_id     = image_name[:-4]

        human_trial_scanpath = utils.get_human_scanpath_for_trial(human_scanpaths, image_name)

        attention_map  = load_model_data(preprocessed_images_dir, img_id, image_size)
        trial_scanpath = create_scanpath_for_trial(trial, attention_map, human_trial_scanpath, image_size, max_fixations, receptive_size, dataset_name)

        scanpaths[image_name] = trial_scanpath
        if trial_scanpath['target_found']:
            targets_found += 1
    
    print('Total targets found: ' + str(targets_found) + '/' + str(len(trials_properties)))
    utils.save_scanpaths(output_path, scanpaths)

def create_scanpath_for_trial(trial, attention_map, human_trial_scanpath, image_size, max_fixations, receptive_size, dataset_name):
    # Load target's boundaries
    target_bbox = (trial['target_matched_row'], trial['target_matched_column'], trial['target_height'] + trial['target_matched_row'], \
        trial['target_width'] + trial['target_matched_column'])
    # Rescale according to image size
    trial_img_size = (trial['image_height'], trial['image_width'])
    target_bbox    = [utils.rescale_coordinate(target_bbox[i], trial_img_size[i % 2 == 1], image_size[i % 2 == 1]) for i in range(len(target_bbox))]
    # Create template of image size, where there are ones in target's box and zeros elsewhere
    # TODO: Cambiar a un mecanismo similar al de IRL y cIBS donde simplemente se fija que esté dentro del target bbox
    target_template = np.zeros(image_size)
    target_template[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]] = 1

    initial_fixation = (trial['initial_fixation_row'], trial['initial_fixation_column'])
    # If executed with --h argument, the model follows human's fixations instead of its own
    human_fixations = []
    if human_trial_scanpath:
        human_fixations  = list(zip(human_trial_scanpath['Y'], human_trial_scanpath['X']))
        initial_fixation = human_fixations[0]
        max_fixations    = len(human_fixations)

    scanpath_x = []
    scanpath_y = []
    target_found = False
    # Compute scanpath from attention map
    for fixation_number in range(max_fixations):
        posY, posX = get_current_fixation(fixation_number, initial_fixation, attention_map, human_fixations)

        scanpath_x.append(int(posX))
        scanpath_y.append(int(posY))

        fixated_window_leftX  = max(posX - receptive_size // 2 + 1, 0)
        fixated_window_leftY  = max(posY - receptive_size // 2 + 1, 0)
        fixated_window_rightX = min(posX + receptive_size // 2, image_size[1])
        fixated_window_rightY = min(posY + receptive_size // 2, image_size[0])

        # Check if target's box overlaps with the fixated window
        fixated_on_target = np.sum(target_template[fixated_window_leftY:fixated_window_rightY, fixated_window_leftX:fixated_window_rightX]) > 0

        if fixated_on_target:
            target_found = True
            break
        else:
            # Apply inhibition of return
            attention_map[fixated_window_leftY:fixated_window_rightY, fixated_window_leftX:fixated_window_rightX] = 0
    
    image_name = trial['image']
    if target_found:
        print(image_name + "; target found at fixation step " + str(fixation_number + 1))
    else:
        print(image_name + "; target NOT FOUND!")

    scanpath = {"subject" : "IVSN Model", "dataset" : dataset_name, "image_height" : image_size[0], "image_width" : image_size[1], "receptive_height" : receptive_size, "receptive_width": receptive_size, \
        "target_found" : target_found, "target_bbox" : target_bbox, "X" : scanpath_x, "Y" : scanpath_y, "target_object" : trial['target_object'], "max_fixations" : max_fixations}
    
    return scanpath

def get_current_fixation(fixation_number, initial_fixation, attention_map, human_fixations):
    if fixation_number == 0:
        posY, posX = initial_fixation
    else:
        if human_fixations:
            posY, posX = human_fixations[fixation_number]
        else:
            max_attention_map = np.where(attention_map == np.amax(attention_map))
            posY, posX  = max_attention_map[0][0], max_attention_map[1][0]
    
    return posY, posX

def load_model_data(preprocessed_images_dir, img_id, image_size):
    # Get attention map for the image
    chopped_img_dir = path.join(preprocessed_images_dir, img_id)
    chopped_data    = listdir(chopped_img_dir)

    template = np.zeros(shape=image_size)
    for chopped_saliency_data in chopped_data:
        if (chopped_saliency_data.endswith('.jpg')):
            continue

        chopped_img_name = chopped_saliency_data[:-18]
        chopped_img      = io.imread(path.join(chopped_img_dir, chopped_img_name + '.jpg'))
        chopped_img_height = chopped_img.shape[0]
        chopped_img_width  = chopped_img.shape[1]
        # Load data computed by the model
        chopped_attention_map = utils.load_dict_from_json(path.join(chopped_img_dir, chopped_saliency_data))
        chopped_attention_map = np.asarray(chopped_attention_map['x'])
        chopped_attention_map = transform.resize(chopped_attention_map, (chopped_img_height, chopped_img_width))
        # Get coordinate information from the chopped image name
        chopped_img_name_split = chopped_img_name.split('_')
        from_row    = int(chopped_img_name_split[len(chopped_img_name_split) - 2])
        from_column = int(chopped_img_name_split[len(chopped_img_name_split) - 1])
        to_row    = from_row    + chopped_img_height
        to_column = from_column + chopped_img_width
        # Replace in template
        template[from_row:to_row, from_column:to_column] = chopped_attention_map

    attention_map = exposure.rescale_intensity(template, out_range=(0, 1))
    
    return attention_map