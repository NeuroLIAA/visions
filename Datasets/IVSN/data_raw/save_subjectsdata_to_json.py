from scipy.io import loadmat
import numpy as np
import json
from os import listdir, mkdir, path

subjects_files_dir = 'ProcessScanpath_naturaldesign/'
trials_sequenceFile = 'naturaldesign_seq.mat'
save_path = '../human_scanpaths/'
# Load targets locations
with open('../trials_properties.json', 'r') as fp:
    trials_properties = json.load(fp)

def get_trial_for_image_name(trials_properties, image_name):
    for trial in trials_properties:
        if trial['image'] == image_name:
            return trial

def load_filtered_images():
    filtered_images_file = open('filtered_images', 'r')
    filtered_images = []
    for line in filtered_images_file:
        filtered_images.append(line.strip())
    
    filtered_images_file.close()
    return filtered_images

num_images = 240
receptive_size = (45, 45)
window_size = (200, 200)
image_size  = (1024, 1280)
screen_size = (1024, 1280)

trials_sequence = loadmat(trials_sequenceFile)
trials_sequence = trials_sequence['seq'].flatten()
trials_sequence = trials_sequence % num_images

subjects_files = listdir(subjects_files_dir)
targets_found = 0
wrong_targets_found = 0
min_scanpath_length = 2
filtered_images = load_filtered_images()

scanpaths_with_shorter_distance_than_receptive_size = 0
shortest_consecutive_fixations_coord_difference     = (9999, 9999)
for subject_data_file in subjects_files:
    if (not(subject_data_file.endswith('_oracle.mat')) or not(subject_data_file.endswith('.mat'))):
        continue
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("Processing " + subject_data_file)

    subject_number = subject_data_file[4:6]

    current_subject_data = loadmat(path.join(subjects_files_dir, subject_data_file))
    subject_trials = dict()
    stimuli_processed = []
    for trial_number in range(len(trials_sequence)):
        stimuli = trials_sequence[trial_number]
        if (stimuli == 0): stimuli = 240
        # Only first trials are taken into account
        if (stimuli in stimuli_processed):
            continue

        stimuli_processed.append(stimuli)        
        stimuli_name = str(stimuli)
        if stimuli < 10:
            stimuli_name = '00' + stimuli_name
        elif stimuli < 100:
            stimuli_name = '0' + stimuli_name
        image_name = 'img' + str(stimuli_name) + '.jpg'

        if image_name in filtered_images:
            continue

        # Get fixation coordinates for the current trial; minus one, since Python indexes images from zero.
        fix_posX = current_subject_data['FixData']['Fix_posx'][0][0][trial_number][0].flatten() - 1
        fix_posY = current_subject_data['FixData']['Fix_posy'][0][0][trial_number][0].flatten() - 1

        number_of_fixations = len(fix_posX)
        # Skip short scanpaths
        if number_of_fixations < min_scanpath_length:
            continue
        target_found = False
        # TargetFound matrix has only 74 columns
        if number_of_fixations < 74:
            target_found = bool(current_subject_data['FixData']['TargetFound'][0][0][trial_number][(number_of_fixations - 1)])

        # Get target position information
        current_target   = get_trial_for_image_name(trials_properties, image_name)
        target_start_row = current_target['target_matched_row']
        target_start_column = current_target['target_matched_column']
        target_end_row      = target_start_row + current_target['target_height']
        target_end_column   = target_start_column + current_target['target_width']

        target_bounding_box = [target_start_row, target_start_column, target_end_row, target_end_column]

        target_center = (target_start_row + (current_target['target_height'] // 2), target_start_column + (current_target['target_width'] // 2))
        target_lower_row_bound  = target_center[0] - window_size[0]
        target_upper_row_bound  = target_center[0] + window_size[0]
        target_lower_column_bound = target_center[1] - window_size[1]
        target_upper_column_bound = target_center[1] + window_size[1]
        if (target_lower_row_bound < 0): target_lower_row_bound = 0
        if (target_upper_row_bound > image_size[0]): target_upper_row_bound = image_size[0]
        if (target_lower_column_bound < 0): target_lower_column_bound = 0
        if (target_upper_column_bound > image_size[1]): target_upper_column_bound = image_size[1]

        last_fixation_X = fix_posX[number_of_fixations - 1]
        last_fixation_Y = fix_posY[number_of_fixations - 1]
        between_bounds = (target_lower_row_bound <= last_fixation_Y) and (target_upper_row_bound >= last_fixation_Y) and \
                         (target_lower_column_bound <= last_fixation_X) and (target_upper_column_bound >= last_fixation_X)
        if target_found:
            if between_bounds:
                targets_found += 1
            else:
                print("Subject: " + subject_number + "; stimuli: " + str(stimuli) + "; trial: " + str(trial_number) + ". Last fixation doesn't match target's bounds")
                print("Target's bounds: " + str((target_lower_row_bound, target_lower_column_bound, target_upper_row_bound, target_upper_column_bound)) + ". Last fixation: " + str((last_fixation_Y, last_fixation_X)) + '\n')
                wrong_targets_found += 1
                target_found = False
        
        if target_found and number_of_fixations > 1:
            fixations = [fix for fix in zip(fix_posX, fix_posY)]
            distance_between_consecutive_fixations = [np.linalg.norm(np.array(fix_1) - np.array(fix_2)) for fix_1, fix_2 in zip(fixations, fixations[1:])]
            shortest_distance_between_consecutive_fixations = min(distance_between_consecutive_fixations)
            fixation_number = distance_between_consecutive_fixations.index(shortest_distance_between_consecutive_fixations)
            shortest_consecutive_fixations_distance = (abs(fix_posX[fixation_number] - fix_posX[fixation_number + 1]), abs(fix_posY[fixation_number] - fix_posY[fixation_number + 1]))
            if shortest_consecutive_fixations_distance[0] < (receptive_size[0] / 2) and shortest_consecutive_fixations_distance[1] < (receptive_size[1] / 2):
                scanpaths_with_shorter_distance_than_receptive_size += 1
                if shortest_consecutive_fixations_distance[0] < shortest_consecutive_fixations_coord_difference[0] \
                    and shortest_consecutive_fixations_distance[1] < shortest_consecutive_fixations_coord_difference[1]:
                    shortest_consecutive_fixations_coord_difference = shortest_consecutive_fixations_distance
                    print("Subject: " + subject_number + "; stimuli: " + str(stimuli) + "; trial: " + str(trial_number) + ". Fixation numbers: " + str(fixation_number + 1) + " " + str(fixation_number + 2) + ". Coord. difference: " + str(shortest_consecutive_fixations_distance))

        fix_startTime = current_subject_data['FixData']['Fix_starttime'][0][0][trial_number][0].flatten()
        fix_endTime = current_subject_data['FixData']['Fix_time'][0][0][trial_number][0].flatten()
        fix_time = fix_endTime - fix_startTime

        subject_trials[image_name] = { "subject" : subject_number, "dataset" : "IVSN Natural Design Dataset", "image_height" : image_size[0], "image_width" : image_size[1], "screen_height" : screen_size[0], "screen_width" : screen_size[1], "receptive_height" : receptive_size[0], "receptive_width" : receptive_size[1], \
            "target_found" : target_found, "target_bbox" : target_bounding_box, "X" : fix_posX.tolist(), "Y" : fix_posY.tolist(), "T" : fix_time.tolist(), "target_object" : "TBD", "max_fixations" : 80}
        

    subject_save_file = 'subj' + subject_number + '_scanpaths.json'
    if not(path.exists(save_path)):
        mkdir(save_path)

    with open(path.join(save_path, subject_save_file), 'w') as fp:
        json.dump(subject_trials, fp, indent=4)

print("Total targets found: " + str(targets_found) + ". Wrong targets found: " + str(wrong_targets_found))
print("Scanpaths with consecutive fixations in a shorter distance than " + str(receptive_size) + ": " + str(scanpaths_with_shorter_distance_than_receptive_size))
print("Min. coordinate difference in consecutive fixations:" + str(shortest_consecutive_fixations_coord_difference))
