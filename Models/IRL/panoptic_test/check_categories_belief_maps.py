import torch
import numpy as np
import json
import cv2
from run_detectron import run_detectron

HR_path = '../Cocosearch18_trainval/DCBs/HR/'
pth_ext = '.pth.tar'
trials_properties_file = '../../../Datasets/COCOSearch18/trials_properties.json'
images_path = '../../../Datasets/COCOSearch18/images/'
threshold = 0.8

with open(trials_properties_file, 'r') as fp:
    trials_properties = json.load(fp)

number_of_categories_not_found = 0
number_of_extra_categories     = 0
categories_not_found = []
extra_categories     = []
images_with_different_categories = 0
for trial in trials_properties:
    image_name = trial['image']
    task = trial['target_object']
    print('Processing image: ' + image_name + '; task: ' + task)

    hr_file = HR_path + image_name[:-4] + pth_ext
    belief_maps = torch.load(hr_file).numpy()
    
    image = cv2.imread(images_path + image_name)
    image = cv2.resize(image, (512, 320))
    panoptic_seg, segments_info = run_detectron(image)

    # Check for categories detected in detectron not found in the belief maps
    total_categories_in_detectron = 0
    categories_in_detectron = []
    for segment in segments_info:
        category_id = segment['category_id']
        is_thing    = segment['isthing']

        should_be = True
        if is_thing:
            score = segment['score']
            if score < threshold: should_be = False
        else:
            category_id += 80
        
        if should_be and not category_id in categories_in_detectron:
            categories_in_detectron.append(category_id)
            total_categories_in_detectron += 1
            if not np.any(belief_maps[category_id]):
                print('Category not found in belief maps: ' + str(category_id))
                if is_thing: print('Score: ' + str(segment['score']))
                number_of_categories_not_found += 1
                if not category_id in categories_not_found:
                    categories_not_found.append(category_id)

    # Check for categories found in the belief maps not detected by detectron
    total_categories_in_belief_maps = 0
    category_id = 0
    for belief_map in belief_maps:
        if np.any(belief_map):
            total_categories_in_belief_maps += 1
            if not category_id in categories_in_detectron:
                print('Category not found in detectron: ' + str(category_id))
                number_of_extra_categories += 1
                if not category_id in extra_categories:
                    extra_categories.append(category_id)
        category_id += 1
    
    if total_categories_in_detectron != total_categories_in_belief_maps:
        print('Different number of categories in belief maps than in detectron.')
        print('Total categories in detectron (score > ' + str(threshold) + '): ' + str(total_categories_in_detectron))
        print('Total categories in belief maps: ' + str(total_categories_in_belief_maps))
        images_with_different_categories += 1
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')

print('Categories in detectron not found in belief maps: ' + str(categories_not_found))
print('Total number of categories in detectron not found in belief maps: ' + str(number_of_categories_not_found))
print('Categories in belief maps not found in detectron (with score > ' + str(threshold) + '): ' + str(extra_categories))
print('Total number of categories in belief maps not found in detectron (with score > ' + str(threshold) + '): ' + str(number_of_extra_categories))
print('Images with different number of categories: ' + str(images_with_different_categories))