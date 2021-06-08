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
    for belief_map in belief_maps:
        if np.any(belief_map):    
            nested_index = 0    
            for another_belief_map in belief_maps:
                if index == nested_index:
                    continue

                if np.any(another_belief_map):
                    breakpoint()
                    overlap = np.any(another_belief_map[belief_map > 0])
                    if overlap:
                        print(image_name + ': overlap of categories ' + str(nested_index) + ' and ' + str(index))
                nested_index += 1
        index += 1
        
