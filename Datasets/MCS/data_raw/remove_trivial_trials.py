import json
import utils
from os import listdir, path

trials_properties_file = '../trials_properties.json'
dataset_info_file      = '../dataset_info.json'

trials_properties = utils.load_dict_from_json(trials_properties_file)
dataset_info      = utils.load_dict_from_json(dataset_info_file)

human_scanpaths_dir   = path.join('..', dataset_info['scanpaths_dir'])
human_scanpaths_files = listdir(human_scanpaths_dir)

receptive_size = dataset_info['receptive_size']

removed_images = []
for trial in list(trials_properties):
    target_bbox      = trial['target_bbox']
    initial_fixation = (trial['initial_fixation_row'], trial['initial_fixation_column'])

    if utils.between_bounds(target_bbox, initial_fixation[0], initial_fixation[1], receptive_size):
        removed_images.append(trial['image'])
        trials_properties.remove(trial)

trials_properties_file = '../trials_properties_cropped.json'
utils.save_to_json(trials_properties_file, trials_properties)

for subject in human_scanpaths_files:
    subject_scanpaths = utils.load_dict_from_json(path.join(human_scanpaths_dir, subject))
    for image in removed_images:
        del subject_scanpaths[image]
    
    utils.save_to_json(path.join(human_scanpaths_dir, subject), subject_scanpaths)

print('Removed images: ' + str(removed_images))
print('Number of removed images: ' + len(removed_images))
