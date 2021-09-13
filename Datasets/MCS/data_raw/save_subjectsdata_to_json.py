import json
from os import listdir, path, mkdir
from utils import rename_image, rescale_coordinate
import pandas as pd
import numpy as np

train_microwave_scanpaths_file = 'gaze/train/microwave_fixations.csv'
train_clock_scanpaths_file     = 'gaze/train/clock_fixations.csv'
test_scanpaths_file            = 'gaze/test/clock_and_microwave_fixations.csv'

targets_bboxes_file = 'targets_bboxes.json'
with open(targets_bboxes_file, 'r') as fp:
    targets_bboxes = json.load(fp)

display_size_train = (800, 1280)
display_size_test  = (1050, 1680)
new_image_size     = (508, 564)
# Estimated by measuring the distance between consecutive fixations
receptive_height = 20
receptive_width  = 20
# Unlimited
max_fixations = 80

save_path = '../human_scanpaths/'

train_microwave_scanpaths = pd.read_csv(train_microwave_scanpaths_file)
train_clock_scanpaths     = pd.read_csv(train_clock_scanpaths_file)
test_scanpaths            = pd.read_csv(test_scanpaths_file)
train_scanpaths           = pd.concat([train_microwave_scanpaths, train_clock_scanpaths])

subjects_scanpaths = {}
number_of_targets_found = 0
number_of_trials = 0

last_trial_processed = 0
subject_num = 0
image_name  = ''
trial_scanpath = {}
rescaled_target_bbox = []
image_category = ''
target_found = False
# Process train scanpaths
for index, row in train_scanpaths.iterrows():
    if row['condition'] == 'absent':
        continue

    trial_index = row['TRIAL_INDEX']
    # Ignore the rest of the scanpath if a previous fixation has landed on the target
    if target_found and last_trial_processed == trial_index:
        continue

    if last_trial_processed != trial_index:
        if trial_scanpath:
            image_name = rename_image(image_name, image_category)
            number_of_trials += 1
            if target_found:
                number_of_targets_found += 1
            if subject_num in subjects_scanpaths:
                subjects_scanpaths[subject_num][image_name] = trial_scanpath
            else:
                subjects_scanpaths[subject_num] = {image_name : trial_scanpath}

        trial_scanpath = {}
        target_found   = False
        last_trial_processed = trial_index
        subject_num = row['subjectnum']
        image_name  = row['searcharray'][15:]
        category    = row['catcue']
        image_category = category

        # Ignore trials of unanotated images
        if not image_name in targets_bboxes[category]:
            target_found = True
            continue
        img_info = targets_bboxes[category][image_name]
        original_img_size = (img_info['image_height'], img_info['image_width'])
        target_bbox = img_info['target_bbox']
        rescaled_target_bbox = [int(rescale_coordinate(target_bbox[i], original_img_size[i % 2 == 1], new_image_size[i % 2 == 1])) for i in range(len(target_bbox))]
        # Edge cases
        rescaled_target_bbox[2] = min(new_image_size[0] - 1, rescaled_target_bbox[2])
        rescaled_target_bbox[3] = min(new_image_size[1] - 1, rescaled_target_bbox[3])

        if subject_num < 10:
            subject_name = '0' + str(subject_num)
        trial_scanpath = {'subject': subject_name, 'dataset': 'MCS Dataset', 'image_height': new_image_size[0], 'image_width': new_image_size[1], \
            'screen_height': display_size_train[0], 'screen_width': display_size_train[1],  'receptive_height': 10, 'receptive_width': 10, 'target_found': False, \
                'target_bbox': rescaled_target_bbox, 'X': [], 'Y': [], 'T': [], 'target_object': category, 'max_fixations': max_fixations}

    current_fix_x = rescale_coordinate(row['CURRENT_FIX_X'], display_size_train[1], new_image_size[1])
    current_fix_y = rescale_coordinate(row['CURRENT_FIX_Y'], display_size_train[0], new_image_size[0])

    trial_scanpath['X'].append(current_fix_x)
    trial_scanpath['Y'].append(current_fix_y)
    trial_scanpath['T'].append(row['CURRENT_FIX_DURATION'])

    target_found = (rescaled_target_bbox[0] <= current_fix_y + receptive_height) and (rescaled_target_bbox[2] >= current_fix_y - receptive_height) and \
                    (rescaled_target_bbox[1] <= current_fix_x + receptive_width) and (rescaled_target_bbox[3] >= current_fix_x - receptive_width)

    trial_scanpath['target_found'] = target_found

# Save last record
if trial_scanpath:
    image_name = rename_image(image_name, image_category)
    if subject_num in subjects_scanpaths:
        subjects_scanpaths[subject_num][image_name] = trial_scanpath
    else:
        subjects_scanpaths[subject_num] = {image_name : trial_scanpath}


last_trial_processed = 0
subject_num = 0
image_name  = ''
trial_scanpath = {}
rescaled_target_bbox = []
image_category = ''
target_found = False
# Process test scanpaths
for index, row in test_scanpaths.iterrows():
    if not row['target_present']:
        continue

    trial_index = row['TRIAL_INDEX']
    # Ignore the rest of the scanpath if a previous fixation has landed on the target
    if target_found and last_trial_processed == trial_index:
        continue

    if last_trial_processed != trial_index:
        if trial_scanpath:
            image_name = rename_image(image_name, image_category)
            number_of_trials += 1
            if target_found:
                number_of_targets_found += 1
            if subject_num in subjects_scanpaths:
                subjects_scanpaths[subject_num][image_name] = trial_scanpath
            else:
                subjects_scanpaths[subject_num] = {image_name : trial_scanpath}

        trial_scanpath = {}
        target_found   = False
        last_trial_processed = trial_index
        subject_num = int(row['RECORDING_SESSION_LABEL'][1:-1])
        image_name  = row['image_name'][-16:]
        if row['RECORDING_SESSION_LABEL'][1:3] == 'c':
            category = 'clock'
        else:
            category = 'microwave'
        image_category = category

        # Ignore trials of unanotated images
        if not image_name in targets_bboxes[category]:
            target_found = True
            continue
        img_info = targets_bboxes[category][image_name]
        original_img_size = (img_info['image_height'], img_info['image_width'])
        target_bbox = img_info['target_bbox']
        rescaled_target_bbox = [int(rescale_coordinate(target_bbox[i], original_img_size[i % 2 == 1], new_image_size[i % 2 == 1])) for i in range(len(target_bbox))]

        if subject_num < 10:
            subject_name = '0' + str(subject_num)
        else:
            subject_name = str(subject_num)
        trial_scanpath = {'subject': subject_name, 'dataset': 'MCS Dataset', 'image_height': new_image_size[0], 'image_width': new_image_size[1], \
            'screen_height': display_size_train[0], 'screen_width': display_size_train[1],  'receptive_height': 10, 'receptive_width': 10, 'target_found': False, \
                'target_bbox': rescaled_target_bbox, 'X': [], 'Y': [], 'T': [], 'target_object': category, 'max_fixations': max_fixations}

    current_fix_x = rescale_coordinate(row['CURRENT_FIX_X'], display_size_test[1], new_image_size[1])
    current_fix_y = rescale_coordinate(row['CURRENT_FIX_Y'], display_size_test[0], new_image_size[0])

    trial_scanpath['X'].append(current_fix_x)
    trial_scanpath['Y'].append(current_fix_y)
    trial_scanpath['T'].append(row['CURRENT_FIX_DURATION'])

    target_found = (rescaled_target_bbox[0] <= current_fix_y + receptive_height) and (rescaled_target_bbox[2] >= current_fix_y - receptive_height) and \
                    (rescaled_target_bbox[1] <= current_fix_x + receptive_width) and (rescaled_target_bbox[3] >= current_fix_x - receptive_width)

    trial_scanpath['target_found'] = target_found

# Save last record
if trial_scanpath:
    image_name = rename_image(image_name, image_category)
    if subject_num in subjects_scanpaths:
        subjects_scanpaths[subject_num][image_name] = trial_scanpath
    else:
        subjects_scanpaths[subject_num] = {image_name : trial_scanpath}

if not path.exists(save_path):
    mkdir(save_path)

scanpaths_with_shorter_distance_than_receptive_size = 0

# Save subjects scanpaths as JSON files and estimate the distance between fixations
for subject in subjects_scanpaths:
    if subject < 10:
        subject_name = '0' + str(subject)
    else:
        subject_name = str(subject)
    subject_scanpaths_file = 'subj' + subject_name + '_scanpaths.json'

    subject_scanpaths = subjects_scanpaths[subject]
    for trial in subject_scanpaths:
        scanpath = subject_scanpaths[trial]
        # Check for distance between consecutive fixations
        if scanpath['target_found'] and len(scanpath['X']) > 1:
            fixations = [fix for fix in zip(scanpath['X'], scanpath['Y'])]
            distance_between_consecutive_fixations          = [np.linalg.norm(np.array(fix_1) - np.array(fix_2)) for fix_1, fix_2 in zip(fixations, fixations[1:])]
            shortest_distance_between_consecutive_fixations = min(distance_between_consecutive_fixations)

            fixation_number = distance_between_consecutive_fixations.index(shortest_distance_between_consecutive_fixations)
            shortest_consecutive_fixations_distance = (abs(scanpath['X'][fixation_number] - scanpath['X'][fixation_number + 1]), abs(scanpath['Y'][fixation_number] - scanpath['Y'][fixation_number + 1]))
            if shortest_consecutive_fixations_distance[0] < receptive_height // 2 and shortest_consecutive_fixations_distance[1] < receptive_width // 2:
                scanpaths_with_shorter_distance_than_receptive_size += 1

    with open(save_path + subject_scanpaths_file, 'w') as fp:
        json.dump(subjects_scanpaths[subject], fp, indent=4)

print('Total targets found: ' + str(number_of_targets_found) + '/' + str(number_of_trials) + ' trials')
print('Scanpaths where saccades have shorter distance than ' + str((receptive_height, receptive_width)) + ': ' + str(scanpaths_with_shorter_distance_than_receptive_size))