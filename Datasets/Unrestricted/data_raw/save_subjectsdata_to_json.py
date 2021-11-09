from scipy.io import loadmat
import numpy as np
import json
import utils
from os import listdir, mkdir, path

subjects_files_dir  = 'ProcessScanpath_naturaldesign/'
trials_sequenceFile = 'naturaldesign_seq.mat'
save_path = '../human_scanpaths'

# Load targets locations
trials_properties  = utils.load_dict_from_json('../trials_properties.json')
targets_categories = utils.load_dict_from_json('targets_categories.json')

num_images = 240
receptive_size = (45, 45)
image_size  = (1024, 1280)
screen_size = (1024, 1280)

trials_sequence = loadmat(trials_sequenceFile)
trials_sequence = trials_sequence['seq'].flatten()
trials_sequence = trials_sequence % num_images

subjects_files = listdir(subjects_files_dir)

number_of_trials    = 0
targets_found       = 0
wrong_targets_found = 0
cropped_scanpaths   = 0
cropped_fixations   = 0
collapsed_scanpaths = 0
collapsed_fixations = 0
trivial_scanpaths   = 0

min_scanpath_length = 2

for subject_data_file in subjects_files:
    if subject_data_file.endswith('_oracle.mat') or not subject_data_file.endswith('.mat'):
        continue
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("Processing " + subject_data_file)

    subject_id = subject_data_file[4:6]

    current_subject_data = loadmat(path.join(subjects_files_dir, subject_data_file))
    subject_trials = dict()
    stimuli_processed = []
    for trial_number in range(len(trials_sequence)):
        stimuli = trials_sequence[trial_number]
        if stimuli == 0: stimuli = 240
        # Only first trials are taken into account
        if stimuli in stimuli_processed:
            continue

        stimuli_processed.append(stimuli)        
        stimuli_name = str(stimuli)
        if stimuli < 10:
            stimuli_name = '00' + stimuli_name
        elif stimuli < 100:
            stimuli_name = '0' + stimuli_name
        image_name = 'img' + str(stimuli_name) + '.jpg'

        # Get fixation coordinates for the current trial; minus one, since Python indexes images from zero.
        scanpath_x = current_subject_data['FixData']['Fix_posx'][0][0][trial_number][0].flatten() - 1
        scanpath_y = current_subject_data['FixData']['Fix_posy'][0][0][trial_number][0].flatten() - 1

        number_of_fixations = len(scanpath_x)

        number_of_trials += 1
        marked_target_found = False
        # TargetFound matrix has only 74 columns
        if number_of_fixations < 74:
            marked_target_found = bool(current_subject_data['FixData']['TargetFound'][0][0][trial_number][(number_of_fixations - 1)])

        # Collapse consecutive fixations which are closer than receptive_size / 2
        scanpath_x, scanpath_y = utils.collapse_fixations(scanpath_x, scanpath_y, receptive_size)
        if len(scanpath_x) < number_of_fixations:
            collapsed_scanpaths += 1
            collapsed_fixations += number_of_fixations - len(scanpath_x)

        number_of_fixations = len(scanpath_x)
        # Get target bounding box
        current_target   = utils.get_trial_for_image_name(trials_properties, image_name)
        # Trivial images are filtered, ignore scanpath
        if current_target == None:
            continue
        target_start_row = current_target['target_matched_row']
        target_start_column = current_target['target_matched_column']
        target_end_row      = target_start_row + current_target['target_height']
        target_end_column   = target_start_column + current_target['target_width']

        target_bbox = [target_start_row, target_start_column, target_end_row, target_end_column]

        # Crop scanpaths as soon as a fixation falls between the target's bounding box
        target_found, scanpath_x, scanpath_y = utils.crop_scanpath(scanpath_x, scanpath_y, target_bbox, receptive_size, image_size)
        # Skip trivial scanpaths
        if len(scanpath_x) < min_scanpath_length:
            trivial_scanpaths += 1
            continue
        if target_found: targets_found += 1
        if len(scanpath_x) < number_of_fixations:
            cropped_scanpaths += 1
            cropped_fixations += number_of_fixations - len(scanpath_x)

        last_fixation_X = scanpath_x[len(scanpath_x) - 1]
        last_fixation_Y = scanpath_y[len(scanpath_y) - 1]
        if marked_target_found:
            if not utils.between_bounds(target_bbox, last_fixation_Y, last_fixation_X, receptive_size):
                print("Subject: " + subject_id + "; stimuli: " + image_name + ". Last fixation doesn't match target's bounds")
                print("Target's bounds: " + str(target_bbox) + ". Last fixation: " + str((last_fixation_Y, last_fixation_X)) + '\n')
                wrong_targets_found += 1
                target_found = False

        fix_startTime = current_subject_data['FixData']['Fix_starttime'][0][0][trial_number][0].flatten()
        fix_endTime   = current_subject_data['FixData']['Fix_time'][0][0][trial_number][0].flatten()
        fix_time = fix_endTime - fix_startTime

        target_object = "TBD"
        if image_name in targets_categories:
            target_object = targets_categories[image_name]['target_object']

        scanpath_x = list(map(int, scanpath_x))
        scanpath_y = list(map(int, scanpath_y))

        subject_trials[image_name] = { "subject" : subject_id, "dataset" : "Unrestricted Dataset", "image_height" : image_size[0], "image_width" : image_size[1], \
            "screen_height" : screen_size[0], "screen_width" : screen_size[1], "receptive_height" : receptive_size[0], "receptive_width" : receptive_size[1], \
                "target_found" : target_found, "target_bbox" : target_bbox, "X" : scanpath_x, "Y" : scanpath_y, "T" : fix_time.tolist(), "target_object" : target_object, "max_fixations" : 80}

    subject_save_file = 'subj' + subject_id + '_scanpaths.json'
    if not(path.exists(save_path)):
        mkdir(save_path)

    utils.save_to_json(path.join(save_path, subject_save_file), subject_trials)

print("Total targets found: " + str(targets_found) + "/" + str(number_of_trials) + " trials. Wrong targets found: " + str(wrong_targets_found))
print("Collapsed scanpaths (discretized in size " + str(receptive_size) + ") : " + str(collapsed_scanpaths))
print("Number of fixations collapsed: " + str(collapsed_fixations))
print("Cropped scanpaths (target found earlier): " + str(cropped_scanpaths))
print("Number of fixations cropped: " + str(cropped_fixations))
print("Trivial scanpaths: " + str(trivial_scanpaths))