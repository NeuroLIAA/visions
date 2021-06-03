from skimage import io, transform, img_as_ubyte, color
from os import listdir, path, mkdir, rename
from scipy.io import savemat
import json
from math import floor

stimuliDir = '../Datasets/IVSN_Color/stimuli/'
targetsDir = '../Datasets/IVSN_Color/target_crops/'
stimuliDirOld = '../Datasets/IVSN_Color/stimuli_old/'

stimuli_size = (768, 1024) #para correr en cIBS
trials_properties_file = '../Datasets/IVSN_Color/trials_properties.json'

def main():

    target_positions = { "target_positions" : []}
    initial_fixations = { "initial_fixations" : []}

    if not(path.exists(stimuliDirOld)):
        mkdir(stimuliDirOld)
    if not(path.exists(targetsDir)):
        mkdir(targetsDir)
    with open(trials_properties_file) as fp:
        trials_properties = json.load(fp)
        
    for trial_properties in trials_properties:
        image_file = trial_properties['image']
        if not(path.exists(stimuliDirOld + image_file)):
            rename(stimuliDir + image_file, stimuliDirOld + image_file)
        image_resized = io.imread(stimuliDirOld + image_file)
        #image_resized = color.rgb2gray(image_resized)
        image_resized = transform.resize(image_resized, stimuli_size)
        io.imsave(stimuliDir + image_file, img_as_ubyte(image_resized), check_contrast=False)
        target_bbox = (trial_properties['target_matched_row'], trial_properties['target_matched_column'], trial_properties['target_height'] + trial_properties['target_matched_row'], \
        trial_properties['target_width'] + trial_properties['target_matched_column'])
        target_bbox = rescale_coordinates(target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3], trial_properties['image_height'], trial_properties['image_width'], stimuli_size[0], stimuli_size[1])
        template = image_resized[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]]     
        io.imsave(targetsDir + image_file[:-4] + '_template.jpg', img_as_ubyte(template), check_contrast=False)


def rescale_coordinates(start_row, start_column, end_row, end_column, img_height, img_width, new_img_height, new_img_width):
    rescaled_start_row = round((start_row / img_height) * new_img_height)
    rescaled_start_column = round((start_column / img_width) * new_img_width)
    rescaled_end_row = floor((end_row / img_height) * new_img_height)
    rescaled_end_column = floor((end_column / img_width) * new_img_width)

    return (rescaled_start_row, rescaled_start_column, rescaled_end_row, rescaled_end_column)
    
main()
