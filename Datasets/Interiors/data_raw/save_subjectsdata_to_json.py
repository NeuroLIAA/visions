import json
import utils
from scipy.io import loadmat
from os import listdir, path, mkdir
import numpy as np

subjects_dir = 'sinfo_subj/'
subjects_files = listdir(subjects_dir)
targets_categories = utils.load_dict_from_json('targets_categories.json')
save_path = '../human_scanpaths/'

receptive_size = (32, 32)
max_scanpath_length = 13

number_of_trials    = 0
targets_found       = 0
wrong_targets_found = 0
collapsed_scanpaths = 0
collapsed_fixations = 0
cropped_scanpaths   = 0
cropped_fixations   = 0
truncated_scanpaths = 0
empty_scanpaths     = 0
trivial_scanpaths   = 0

for subject_file in subjects_files:
    subject_info = loadmat(path.join(subjects_dir, subject_file))
    subject_info = subject_info['info_per_subj']

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Processing ' + subject_file)
    print('\n')

    split_subject_filename = subject_file.split('_')
    subject_id = split_subject_filename[len(split_subject_filename) - 1][:-4]
    if (int(subject_id) < 10):
        subject_id = '0' + subject_id

    json_subject = dict()
    for record in range(len(subject_info[0])):
        image_name   = subject_info['image_name'][0][record][0]
        image_height = int(subject_info['image_size'][0][record][0][0])
        image_width  = int(subject_info['image_size'][0][record][0][1])

        screen_height = int(subject_info['screen_size'][0][record][0][0])
        screen_width  = int(subject_info['screen_size'][0][record][0][1])

        target_bbox = subject_info['target_rect'][0][record][0]
        # Swap values, new order is [lower_row, lower_column, upper_row, upper_column]
        target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3] = target_bbox[1] - 1, target_bbox[0] - 1, target_bbox[3] - 1, target_bbox[2] - 1
        marked_target_found = bool(subject_info['target_found'][0][record][0][0])

        trial_max_fixations = int(subject_info['nsaccades_allowed'][0][record][0][0]) + 1

        # Subtract one, since Python indexes images from zero
        scanpath_x = subject_info['x'][0][record][0].astype(float) - 1
        scanpath_y = subject_info['y'][0][record][0].astype(float) - 1
        scanpath_time = subject_info['dur'][0][record][0]

        # Truncate negative values
        scanpath_x = list(np.where(scanpath_x < 0, 0, scanpath_x))
        scanpath_y = list(np.where(scanpath_y < 0, 0, scanpath_y))

        if len(scanpath_x) == 0:
            print("Subject: " + subject_id + "; stimuli: " + image_name + "; trial: " + str(record + 1) + ". Empty scanpath")
            empty_scanpaths += 1
            continue
        
        number_of_trials += 1
        original_scanpath_len = len(scanpath_x)
        # Collapse consecutive fixations which are closer than receptive_size / 2
        scanpath_x, scanpath_y = utils.collapse_fixations(scanpath_x, scanpath_y, receptive_size)
        if len(scanpath_x) < original_scanpath_len:
            collapsed_scanpaths += 1
            collapsed_fixations += original_scanpath_len - len(scanpath_x)

        original_scanpath_len = len(scanpath_x)
        # Crop scanpaths as soon as a fixation falls between the target's bounding box
        target_found, scanpath_x, scanpath_y = utils.crop_scanpath(scanpath_x, scanpath_y, target_bbox, receptive_size, (image_height, image_width))
        # Ignore trivial scanpaths
        if len(scanpath_x) == 1:
            trivial_scanpaths += 1
            continue
        if target_found: targets_found += 1
        if len(scanpath_x) < original_scanpath_len:
            cropped_scanpaths += 1
            cropped_fixations += original_scanpath_len - len(scanpath_x)

        scanpath_length = len(scanpath_x)
        if trial_max_fixations > max_scanpath_length:
            trial_max_fixations = max_scanpath_length
            if scanpath_length > max_scanpath_length:
                # Truncate scanpath
                scanpath_x = scanpath_x[:max_scanpath_length]
                scanpath_y = scanpath_y[:max_scanpath_length]
                scanpath_length = max_scanpath_length
                if target_found:
                    target_found = False
                    targets_found -= 1
                truncated_scanpaths += 1
                    
        last_fixation_X = scanpath_x[scanpath_length - 1]
        last_fixation_Y = scanpath_y[scanpath_length - 1]
        if marked_target_found:
            if not utils.between_bounds(target_bbox, last_fixation_Y, last_fixation_X, receptive_size):
                print("Subject: " + subject_id + "; stimuli: " + image_name + "; trial: " + str(record + 1) + ". Last fixation doesn't match target's bounds")
                print("Target's bounds: " + str(target_bbox) + ". Last fixation: " + str((last_fixation_Y, last_fixation_X)) + '\n')
                wrong_targets_found += 1
                target_found = False
        
        target_object = "TBD"
        if image_name in targets_categories:
            target_object = targets_categories[image_name]['target_object']

        json_subject[image_name] = {"subject" : subject_id, "dataset" : "Interiors Dataset", "image_height" : image_height, "image_width" : image_width, "screen_height" : screen_height, "screen_width" : screen_width, "receptive_height" : receptive_size[0], "receptive_width" : receptive_size[1], \
            "target_found" : target_found, "target_bbox" : target_bbox.tolist(), "X" : scanpath_x, "Y" : scanpath_y, "T" : scanpath_time.tolist(), "target_object" : target_object, "max_fixations" : trial_max_fixations}
    
    if not(path.exists(save_path)):
        mkdir(save_path)

    subject_json_filename = 'subj' + subject_id + '_scanpaths.json'
    utils.save_to_json(path.join(save_path, subject_json_filename), json_subject)

print("Targets found: " + str(targets_found) + "/" + str(number_of_trials) + ". Wrong targets found: " + str(wrong_targets_found))
print("Collapsed scanpaths (discretized in size " + str(receptive_size) + ") : " + str(collapsed_scanpaths))
print("Number of fixations collapsed: " + str(collapsed_fixations))
print("Cropped scanpaths (target found earlier): " + str(cropped_scanpaths))
print("Number of cropped fixations: " + str(cropped_fixations))
print("Truncated scanpaths: " + str(truncated_scanpaths))
print("Empty scanpaths: " + str(empty_scanpaths))
print("Trivial scanpaths: " + str(trivial_scanpaths))
