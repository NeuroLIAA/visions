import json
import os

scanpaths_file = 'Scanpaths.json'
with open(scanpaths_file, 'r') as fp:
    scanpaths = json.load(fp)

receptive_height = 150
receptive_width  = 160
targets_found = 0

for trial in scanpaths:
    scanpath_info = scanpaths[trial]
    scanpath_x = scanpath_info['X']
    scanpath_y = scanpath_info['Y']
    target_bbox = scanpath_info['target_bbox']

    target_center = (target_bbox[0] + ((target_bbox[2] - target_bbox[0]) // 2), target_bbox[1] + ((target_bbox[3] - target_bbox[1]) // 2))
    target_lower_row_bound = target_center[0] - receptive_height
    target_upper_row_bound = target_center[0] + receptive_height
    target_lower_column_bound = target_center[1] - receptive_width
    target_upper_column_bound = target_center[1] + receptive_width

    original_scanpath_length = len(scanpath_x)
    target_found = False
    scanpath = zip(scanpath_x, scanpath_y)
    fix_number = 1
    for fixation in scanpath:
        # Check if fixation falls between target's window bounds
        if fixation[1] >= target_lower_row_bound and fixation[1] <= target_upper_row_bound and \
            fixation[0] >= target_lower_column_bound and fixation[0] <= target_upper_column_bound:
            target_found = True
            targets_found += 1
            if fix_number < original_scanpath_length:
                print(trial + ' target found earlier')
            break
        
        fix_number += 1
    
    scanpath_info['X'] = scanpath_x[:fix_number]
    scanpath_info['Y'] = scanpath_y[:fix_number]
    scanpath_info['target_found'] = target_found

print('Targets found: ' + str(targets_found))
with open('Scanpaths_as_humans.json', 'w') as fp:
    json.dump(scanpaths, fp, indent=4)