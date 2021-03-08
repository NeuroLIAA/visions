from skimage import io, transform, img_as_ubyte, color
from os import listdir, path, mkdir, rename
from scipy.io import savemat
import json

datasetDir = '../data_images/images/'
targetsDir = '../data_images/templates/'
datasetDirOld = '../data_images/images_old/'
targetsDirOld = '../data_images/templates_old/'
stimuli_size = (768, 1024)
trials_properties_file = '../data_images/trials_properties_IVSN.json'

def main():

    target_positions = { "target_positions" : []}
    initial_fixations = { "initial_fixations" : []}

    if not(path.exists(datasetDirOld)):
        mkdir(datasetDirOld)
    if not(path.exists(targetsDirOld)):
        mkdir(targetsDirOld)
    with open(trials_properties_file) as fp:
        trials_properties = json.load(fp)
    
    for template_file in listdir(targetsDir):
        if not(template_file.endswith('.jpg')):
            continue
        rename(targetsDir + template_file, targetsDirOld + template_file)

    for trial_properties in trials_properties:
        image_file = trial_properties['image']
        rename(datasetDir + image_file, datasetDirOld + image_file)
        image_resized = io.imread(datasetDirOld + image_file)
        image_resized = color.rgb2gray(image_resized)
        image_resized = transform.resize(image_resized, stimuli_size)
        io.imsave(datasetDir + image_file, img_as_ubyte(image_resized), check_contrast=False)
        target_bbox = (trial_properties['target_matched_row'], trial_properties['target_matched_column'], trial_properties['target_side_length'] + trial_properties['target_matched_row'], \
        trial_properties['target_columns'] + trial_properties['target_matched_column'])
        target_bbox = rescale_coordinates(target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3], trial_properties['image_height'], trial_properties['image_width'], stimuli_size[0], stimuli_size[1])
        template = image_resized[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]]       
        io.imsave(targetsDir + image_file[:-4] + '_template.jpg', img_as_ubyte(template), check_contrast=False)
        target_positions["target_positions"].append({ "image" : image_file, "template" : image_file[:-4] + '_template.jpg', "matched_column" : target_bbox[1] + 1, "matched_row" : target_bbox[0] + 1, "template_side_length" : target_bbox[2] - target_bbox[0], "template_columns" : target_bbox[3] - target_bbox[1] }) #side length y columns son distancias, no hace falta sumar 1
        initial_fixations["initial_fixations"].append({"image" : image_file, "initial_fix" : (trial_properties['initial_fixation_x'],trial_properties['initial_fixation_y'])})
    savemat("target_positions_filtered.mat", target_positions)
    savemat("initial_fixations.mat", initial_fixations)

def rescale_coordinates(start_row, start_column, end_row, end_column, img_height, img_width, new_img_height, new_img_width):
    rescaled_start_row = round((start_row / img_height) * new_img_height)
    rescaled_start_column = round((start_column / img_width) * new_img_width)
    rescaled_end_row = round((end_row / img_height) * new_img_height)
    rescaled_end_column = round((end_column / img_width) * new_img_width)

    return rescaled_start_row, rescaled_start_column, rescaled_end_row, rescaled_end_column
    
main()
