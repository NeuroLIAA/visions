import torch
import numpy as np
import json

HR_path = '../Cocosearch18_trainval/DCBs/HR/'
pth_ext = '.pth.tar'
trials_properties_file = '../../../Datasets/COCOSearch18/trials_properties.json'

with open(trials_properties_file, 'r') as fp:
    trials_properties = json.load(fp)

for trial in trials_properties:
    image_name = trial['image']

    hr_file = HR_path + image_name[:-4] + pth_ext
    belief_maps = torch.load(hr_file).numpy()

    index = 0
    complete_matrix = np.zeros(shape=belief_maps.shape[1:3])
    for belief_map in belief_maps:
        if np.any(belief_map):    
            true_values = np.where(belief_map > 0)
            true_coords = list(zip(true_values[0], true_values[1]))
            for coord in true_coords:
                complete_matrix[coord[0], coord[1]] = 1

            nested_index = 0    
            for another_belief_map in belief_maps:
                if index == nested_index:
                    continue

                if np.any(another_belief_map):
                    overlap = np.any(another_belief_map[belief_map > 0])
                    if overlap:
                        print(image_name + ': overlap of categories ' + str(nested_index) + ' and ' + str(index))

                nested_index += 1
        index += 1
    
    if np.all(complete_matrix):
        print('Image: ' + image_name + ' has a category for every pixel')
        
