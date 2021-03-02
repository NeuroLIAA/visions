import json
import numpy as np
from os import listdir, mkdir, path
from skimage import io, transform, exposure

"""
Puts together data produced by the CNN and creates an attention map for the stimuli, which is used to compute the scanpaths, with a winner-takes-all strategy.
Scanpaths are saved in a JSON file.
"""

def parse_model_data(stimuli_dir, chopped_dir, stimuli_size, max_fixations, receptive_size, save_path, trials_properties, dataset_name):
    scanpaths = dict()
    for trial_properties in trials_properties:
        imageName = trial_properties['image']

        imgID = imageName[:-4]
        attentionMap = load_model_data(chopped_dir, imgID, stimuli_size)
        create_scanpath(trial_properties, attentionMap, stimuli_size, max_fixations, receptive_size, dataset_name, scanpaths)
    
    if not(path.exists(save_path)):
        mkdir(save_path)
    with open(save_path + 'Scanpaths.json', 'w') as json_file:
        json.dump(scanpaths, json_file, indent = 4)

def create_scanpath(trial_properties, attentionMap, stimuli_size, max_fixations, receptive_size, dataset_name, scanpaths):
    # Load target's boundaries
    target_bbox = (trial_properties['target_matched_row'], trial_properties['target_matched_column'], trial_properties['target_side_length'] + trial_properties['target_matched_row'], \
        trial_properties['target_columns'] + trial_properties['target_matched_column'])
    # Rescale according to stimuli size
    target_bbox = rescale_coordinates(target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3], trial_properties['image_height'], trial_properties['image_width'], stimuli_size[0], stimuli_size[1])
    # Create template of stimuli's size where there are ones in target's box and zeros elsewhere
    targetTemplate = np.zeros(stimuli_size)
    targetTemplate[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]] = 1

    scanpath_x_coordinates = []
    scanpath_y_coordinates = []
    target_found = False
    first_fixation = True
    # Compute scanpaths from saliency image        
    for fixationNumber in range(max_fixations):
        if first_fixation:
            posX = trial_properties['initial_fixation_x']
            posY = trial_properties['initial_fixation_y']
        else:
            coordinates = np.where(attentionMap == np.amax(attentionMap))
            posX = coordinates[0][0]
            posY = coordinates[1][0]

        scanpath_x_coordinates.append(int(posX))
        scanpath_y_coordinates.append(int(posY))

        fixatedPlace_leftX  = posX - receptive_size // 2 + 1
        fixatedPlace_rightX = posX + receptive_size // 2
        fixatedPlace_leftY  = posY - receptive_size // 2 + 1
        fixatedPlace_rightY = posY + receptive_size // 2

        if fixatedPlace_leftX < 0: fixatedPlace_leftX = 0
        if fixatedPlace_leftY < 0: fixatedPlace_leftY = 0
        if fixatedPlace_rightX > stimuli_size[0]: fixatedPlace_rightX = stimuli_size[0]
        if fixatedPlace_rightY > stimuli_size[1]: fixatedPlace_rightY = stimuli_size[1]

        # Check if target's box overlaps with the fixated place
        fixatedPlace = targetTemplate[fixatedPlace_leftX:fixatedPlace_rightX, fixatedPlace_leftY:fixatedPlace_rightY]

        if (np.sum(fixatedPlace) > 0):
            target_found = True
            break
        else:
            # Apply inhibition of return
            attentionMap[fixatedPlace_leftX:fixatedPlace_rightX, fixatedPlace_leftY:fixatedPlace_rightY] = 0
            
        first_fixation = False
    
    imageName = trial_properties['image']
    if (target_found):
        print(imageName + "; target found at fixation step " + str(fixationNumber + 1))
    else:
        print(imageName + "; target NOT FOUND!")
    scanpaths[imageName] = { "dataset" : dataset_name, "subject" : "IVSN Model", "target_found"  : target_found, "X" : scanpath_x_coordinates, "Y" : scanpath_y_coordinates,  \
        "image_height" : stimuli_size[0], "image_width" : stimuli_size[1], "target_object" : "TBD", "max_fixations" : max_fixations}


def rescale_coordinates(start_row, start_column, end_row, end_column, img_height, img_width, new_img_height, new_img_width):
    rescaled_start_row = round((start_row / img_height) * new_img_height)
    rescaled_start_column = round((start_column / img_width) * new_img_width)
    rescaled_end_row = round((end_row / img_height) * new_img_height)
    rescaled_end_column = round((end_column / img_width) * new_img_width)

    return rescaled_start_row, rescaled_start_column, rescaled_end_row, rescaled_end_column

def load_model_data(chopped_dir, imgID, stimuli_size):
    # Get attention map for stimuli
    choppedImgDir = chopped_dir + imgID + '/'
    choppedData = listdir(choppedImgDir)

    template = np.zeros([stimuli_size[0], stimuli_size[1]])
    for choppedSaliencyData in choppedData:
        if (choppedSaliencyData.endswith('.jpg')):
            continue

        choppedImgName = choppedSaliencyData[:-18]
        choppedImg     = io.imread(choppedImgDir + choppedImgName + '.jpg')
        choppedImg_height = choppedImg.shape[0]
        choppedImg_width  = choppedImg.shape[1]
        # Load data computed by the model
        with open(choppedImgDir + choppedSaliencyData, 'r') as json_attention_map:
            choppedAttentionMap = json.load(json_attention_map)
        choppedAttentionMap = np.asarray(choppedAttentionMap['x'])
        choppedAttentionMap = transform.resize(choppedAttentionMap, (choppedImg_height, choppedImg_width))
        # Get coordinate information from the chopped image name
        choppedImgNameSplit = choppedImgName.split('_')
        from_row    = int(choppedImgNameSplit[len(choppedImgNameSplit) - 2])
        from_column = int(choppedImgNameSplit[len(choppedImgNameSplit) - 1])
        to_row    = from_row + choppedImg_height
        to_column = from_column + choppedImg_width
        # Replace in template
        template[from_row:to_row, from_column:to_column] = choppedAttentionMap
    attentionMap = exposure.rescale_intensity(template, out_range=(0, 1))
    
    return attentionMap
