import json
import numpy as np
from os import listdir, makedirs, path
from skimage import io, transform, exposure

"""
Puts together data produced by the CNN and creates an attention map for the image, which is used to compute the scanpaths, with a winner-takes-all strategy.
Scanpaths are saved in a JSON file.
"""

def parse_model_data(preprocessed_images_dir, trials_properties, image_size, max_fixations, receptive_size, dataset_name, output_path):
    scanpaths = dict()
    for trial_properties in trials_properties:
        image_name = trial_properties['image']
        img_id     = image_name[:-4]

        attention_map = load_model_data(preprocessed_images_dir, img_id, image_size)

        create_scanpath(trial_properties, attention_map, image_size, max_fixations, receptive_size, dataset_name, scanpaths)
    
    if not(path.exists(output_path)):
        makedirs(output_path)
    with open(output_path + 'Scanpaths.json', 'w') as json_file:
        json.dump(scanpaths, json_file, indent = 4)

def create_scanpath(trial_properties, attention_map, image_size, max_fixations, receptive_size, dataset_name, scanpaths):
    # Load target's boundaries
    target_bbox = (trial_properties['target_matched_row'], trial_properties['target_matched_column'], trial_properties['target_height'] + trial_properties['target_matched_row'], \
        trial_properties['target_width'] + trial_properties['target_matched_column'])
    # Rescale according to stimuli size
    target_bbox = rescale_coordinates(target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3], trial_properties['image_height'], trial_properties['image_width'], image_size[0], image_size[1])
    # Create template of stimuli's size where there are ones in target's box and zeros elsewhere
    target_template = np.zeros(image_size)
    target_template[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]] = 1

    scanpath_x = []
    scanpath_y = []
    target_found   = False
    # Compute scanpaths from saliency image        
    for fixation_number in range(max_fixations):
        if fixation_number == 0:
            posY = trial_properties['initial_fixation_row']
            posX = trial_properties['initial_fixation_column']
        else:
            coordinates = np.where(attention_map == np.amax(attention_map))
            posY = coordinates[0][0]
            posX = coordinates[1][0]

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
    
    image_name = trial_properties['image']
    if (target_found):
        print(image_name + "; target found at fixation step " + str(fixation_number + 1))
    else:
        print(image_name + "; target NOT FOUND!")

    scanpaths[image_name] = { "subject" : "IVSN Model", "dataset" : dataset_name, "image_height" : image_size[0], "image_width" : image_size[1], "receptive_height" : receptive_size, "receptive_width": receptive_size, \
        "target_found" : target_found, "target_bbox" : target_bbox, "X" : scanpath_x, "Y" : scanpath_y, "target_object" : trial_properties['target_object'], "max_fixations" : max_fixations}


def rescale_coordinates(start_row, start_column, end_row, end_column, img_height, img_width, new_img_height, new_img_width):
    rescaled_start_row = round((start_row / img_height) * new_img_height)
    rescaled_start_column = round((start_column / img_width) * new_img_width)
    rescaled_end_row = round((end_row / img_height) * new_img_height)
    rescaled_end_column = round((end_column / img_width) * new_img_width)

    return rescaled_start_row, rescaled_start_column, rescaled_end_row, rescaled_end_column

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
        with open(chopped_img_dir + chopped_saliency_data, 'r') as json_attention_map:
            chopped_attention_map = json.load(json_attention_map)
        chopped_attention_map = np.asarray(chopped_attention_map['x'])
        chopped_attention_map = transform.resize(chopped_attention_map, (chopped_img_height, chopped_img_width))
        # Get coordinate information from the chopped image name
        chopped_img_name_split = chopped_img_name.split('_')
        from_row    = int(chopped_img_name_split[len(chopped_img_name_split) - 2])
        from_column = int(chopped_img_name_split[len(chopped_img_name_split) - 1])
        to_row    = from_row + chopped_img_height
        to_column = from_column + chopped_img_width
        # Replace in template
        template[from_row:to_row, from_column:to_column] = chopped_attention_map
    attention_map = exposure.rescale_intensity(template, out_range=(0, 1))
    
    return attention_map
