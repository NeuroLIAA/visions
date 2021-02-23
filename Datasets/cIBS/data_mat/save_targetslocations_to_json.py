from scipy.io import loadmat
import json
from os import listdir, mkdir, path

targets_positions_file = 'target_positions_filtered.mat'
targets_positions_mat = loadmat(targets_positions_file)
targets_positions_mat = targets_positions_mat['target_positions']

image_height = 768
image_width = 1024

targets_positions_json = []
for record in range(len(targets_positions_mat[0])):
    image_name = targets_positions_mat['image'][0][record][0]
    template_name = targets_positions_mat['template'][0][record][0]
    matched_row = int(targets_positions_mat['matched_row'][0][record][0][0])
    matched_column = int(targets_positions_mat['matched_column'][0][record][0][0])

    target_side_length = int(targets_positions_mat['template_side_length'][0][record][0][0])
    target_columns = int(targets_positions_mat['template_columns'][0][record][0][0])


    targets_positions_json.append({ "image" : image_name, "template" : template_name, "dataset" : "cIBS Dataset", "matched_row" : matched_row, "matched_column" : matched_column, \
         "target_side_length" : target_side_length, "target_columns" : target_columns, "image_height" : image_height, "image_width" : image_width})

with open('../targets_locations.json', 'w') as fp:
    json.dump(targets_positions_json, fp, indent=4)