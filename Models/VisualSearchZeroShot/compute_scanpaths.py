import json
import numpy as np
from os import listdir
from skimage import io, transform, exposure

"""
Puts together data produced by the CNN and creates an attention map for the stimuli, which is used to compute the scanpath, with a winner-takes-all strategy.
Scanpaths are saved in a JSON file.
"""

def parse_model_data(stimuli_dir, chopped_dir, stimuli_size, max_fixations, receptive_size, save_path, targets_locations):
    scanpaths = []
    for struct in targets_locations:
        imageName = struct['image']

        imgID = imageName[:-4]
        attentionMap = load_model_data(chopped_dir, imgID, stimuli_size)
        create_scanpath(scanpaths, struct, attentionMap, targets_locations, stimuli_size, max_fixations, receptive_size)
    
    with open(save_path, 'w') as json_file:
        json.dump(scanpaths, json_file, indent = 4)

def create_scanpath(scanpaths, target_properties, attentionMap, targets_locations, stimuli_size, max_fixations, receptive_size):
    # Load target's boundaries
    target_bbox = (target_properties['matched_row'], target_properties['matched_column'], target_properties['target_side_length'] + target_properties['matched_row'], \
        target_properties['target_columns'] + target_properties['matched_column'])
    # Rescale according to stimuli size
    target_bbox = rescale_coordinates(target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3], target_properties['image_height'], target_properties['image_width'], stimuli_size[0], stimuli_size[1])
    # Create template of stimuli's size where there are ones in target's box and zeros elsewhere
    targetTemplate = np.zeros(stimuli_size)
    targetTemplate[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]] = 1

    xCoordFixationOrder = []
    yCoordFixationOrder = []
    target_found = False
    # Compute scanpaths from saliency image        
    for fixationNumber in range(max_fixations):
        coordinates = np.where(attentionMap == np.amax(attentionMap))
        posX = coordinates[0][0]
        posY = coordinates[1][0]

        xCoordFixationOrder.append(str(posX))
        yCoordFixationOrder.append(str(posY))

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
    
    imageName = target_properties['image']
    if (target_found):
        print(imageName + "; target found at fixation step " + str(fixationNumber + 1))
    else:
        print(imageName + "; target NOT FOUND!")
    scanpaths.append({ "image" : imageName, "dataset" : "VisualSearchZeroShot Natural Design Dataset", "subject" : "VisualSearchZeroShot Model", "target_found"  : str(target_found), "X" : xCoordFixationOrder, "Y" : yCoordFixationOrder,  "split" : "test", \
        "image_height" : stimuli_size[0], "image_width" : stimuli_size[1], "target_object" : "TBD", "max_fixations" : str(max_fixations)})


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
