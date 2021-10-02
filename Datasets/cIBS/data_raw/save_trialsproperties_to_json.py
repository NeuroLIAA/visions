from os import listdir, path
from scipy.io import loadmat
import utils

targets_positions_file = 'target_positions_filtered.mat'
targets_positions_mat = loadmat(targets_positions_file)
targets_positions_mat = targets_positions_mat['target_positions']

initial_fixations_file = 'initial_fixations.mat'
initial_fixations_mat = loadmat(initial_fixations_file)
initial_fixations_mat = initial_fixations_mat['initial_fixations']

targets_categories = utils.load_dict_from_json('targets_categories.json')

image_height = 768
image_width  = 1024

trials_properties = []
for record in range(len(targets_positions_mat[0])):
    image_name = targets_positions_mat['image'][0][record][0]
    template_name = targets_positions_mat['template'][0][record][0]
    # Subtract one, since Python indexes from zero
    matched_row = int(targets_positions_mat['matched_row'][0][record][0][0]) - 1
    matched_column = int(targets_positions_mat['matched_column'][0][record][0][0]) - 1

    target_height = int(targets_positions_mat['template_side_length'][0][record][0][0])
    target_width  = int(targets_positions_mat['template_columns'][0][record][0][0])

    initial_fixation_row    = int(initial_fixations_mat['initial_fix'][0][record][0][0]) - 1
    initial_fixation_column = int(initial_fixations_mat['initial_fix'][0][record][0][1]) - 1

    target_object = "TBD"
    if image_name in targets_categories:
        target_object = targets_categories[image_name]['target_object']

    trials_properties.append({ "image" : image_name, "target" : template_name, "dataset" : "cIBS Dataset", "target_matched_row" : matched_row, "target_matched_column" : matched_column, \
         "target_height" : target_height, "target_width" : target_width, "image_height" : image_height, "image_width" : image_width, "initial_fixation_row" : initial_fixation_row, "initial_fixation_column" : initial_fixation_column, \
         "target_object" : target_object})

trials_properties = utils.sorted_alphanumeric(trials_properties)

utils.save_to_json('../trials_properties.json', trials_properties)
