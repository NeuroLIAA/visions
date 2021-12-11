import json
from os import listdir, path, mkdir
import utils
import pandas as pd
import numpy as np

train_microwave_scanpaths_file = 'gaze/train/microwave_fixations.csv'
train_clock_scanpaths_file     = 'gaze/train/clock_fixations.csv'
test_scanpaths_file            = 'gaze/test/clock_and_microwave_fixations.csv'

targets_bboxes_file = 'targets_bboxes.json'
targets_bboxes      = utils.load_dict_from_json(targets_bboxes_file)

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
number_of_trials = 0

last_trial_processed = 0
subject_num = 0
image_name  = ''
trial_scanpath = {}
rescaled_target_bbox = []
image_category = ''
target_found = False
unanotated   = False
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
            image_name = utils.rename_image(image_name, image_category)
            number_of_trials += 1
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
        rescaled_target_bbox = [int(utils.rescale_coordinate(target_bbox[i], original_img_size[i % 2 == 1], new_image_size[i % 2 == 1])) for i in range(len(target_bbox))]
        # Edge cases
        rescaled_target_bbox[2] = min(new_image_size[0] - 1, rescaled_target_bbox[2])
        rescaled_target_bbox[3] = min(new_image_size[1] - 1, rescaled_target_bbox[3])

        if subject_num < 10:
            subject_name = '0' + str(subject_num)

        trial_scanpath = {'subject': subject_name, 'dataset': 'MCS Dataset', 'image_height': new_image_size[0], 'image_width': new_image_size[1], \
            'screen_height': display_size_train[0], 'screen_width': display_size_train[1],  'receptive_height': receptive_height, 'receptive_width': receptive_width, 'target_found': False, \
                'target_bbox': rescaled_target_bbox, 'X': [], 'Y': [], 'T': [], 'target_object': category, 'max_fixations': max_fixations}

    current_fix_x, current_fix_y = utils.convert_coordinate(row['CURRENT_FIX_X'], row['CURRENT_FIX_Y'], original_img_size[1], original_img_size[0], display_size_train, new_image_size, is_train=True)

    trial_scanpath['X'].append(current_fix_x)
    trial_scanpath['Y'].append(current_fix_y)
    trial_scanpath['T'].append(row['CURRENT_FIX_DURATION'])

    target_found = utils.between_bounds(rescaled_target_bbox, current_fix_y, current_fix_x, (receptive_height, receptive_width))

    trial_scanpath['target_found'] = target_found

# Save last record
if trial_scanpath:
    image_name = utils.rename_image(image_name, image_category)
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
            image_name = utils.rename_image(image_name, image_category)
            number_of_trials += 1
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
        rescaled_target_bbox = [int(utils.rescale_coordinate(target_bbox[i], original_img_size[i % 2 == 1], new_image_size[i % 2 == 1])) for i in range(len(target_bbox))]

        if subject_num < 10:
            subject_name = '0' + str(subject_num)
        else:
            subject_name = str(subject_num)
        trial_scanpath = {'subject': subject_name, 'dataset': 'MCS Dataset', 'image_height': new_image_size[0], 'image_width': new_image_size[1], \
            'screen_height': display_size_train[0], 'screen_width': display_size_train[1],  'receptive_height': receptive_height, 'receptive_width': receptive_width, 'target_found': False, \
                'target_bbox': rescaled_target_bbox, 'X': [], 'Y': [], 'T': [], 'target_object': category, 'max_fixations': max_fixations}

    current_fix_x, current_fix_y = utils.convert_coordinate(row['CURRENT_FIX_X'], row['CURRENT_FIX_Y'], original_img_size[1], original_img_size[0], display_size_test, new_image_size, is_train=False)

    trial_scanpath['X'].append(current_fix_x)
    trial_scanpath['Y'].append(current_fix_y)
    trial_scanpath['T'].append(row['CURRENT_FIX_DURATION'])

    target_found = utils.between_bounds(rescaled_target_bbox, current_fix_y, current_fix_x, (receptive_height, receptive_width))

    trial_scanpath['target_found'] = target_found

# Save last record
if trial_scanpath:
    image_name = utils.rename_image(image_name, image_category)
    if subject_num in subjects_scanpaths:
        subjects_scanpaths[subject_num][image_name] = trial_scanpath
    else:
        subjects_scanpaths[subject_num] = {image_name : trial_scanpath}

if not path.exists(save_path):
    mkdir(save_path)

# Save subjects scanpaths as JSON files and collapse fixations
collapsed_scanpaths = 0
collapsed_fixations = 0
targets_found       = 0
for subject in subjects_scanpaths:
    if subject < 10:
        subject_name = '0' + str(subject)
    else:
        subject_name = str(subject)
    subject_scanpaths_file = 'subj' + subject_name + '_scanpaths.json'

    subject_scanpaths = subjects_scanpaths[subject]

    for trial in subject_scanpaths:
        scanpath = subject_scanpaths[trial]
        original_scanpath_length = len(scanpath['X'])
        targets_found += scanpath['target_found']
        # Collapse consecutive fixations which are closer than receptive_size / 2
        scanpath['X'], scanpath['Y'] = utils.collapse_fixations(scanpath['X'], scanpath['Y'], (receptive_height, receptive_width))
        if len(scanpath['X']) < original_scanpath_length:
            collapsed_scanpaths += 1
            collapsed_fixations += original_scanpath_length - len(scanpath['X'])

    utils.save_to_json(path.join(save_path, subject_scanpaths_file), subjects_scanpaths[subject])

print('Total targets found: ' + str(targets_found) + '/' + str(number_of_trials) + ' trials')
print("Collapsed scanpaths (discretized in size " + str((receptive_height, receptive_width)) + ") : " + str(collapsed_scanpaths))
print("Number of fixations collapsed: " + str(collapsed_fixations))