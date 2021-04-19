from scipy.io import loadmat
import numpy as np
import json
from os import listdir, mkdir, path

subjectsFilesDir = 'ProcessScanpath_naturaldesign/'
trialsSequenceFile = 'naturaldesign_seq.mat'
save_path = '../human_scanpaths/'
# Load targets locations
targetsPropertiesFile = open('../trials_properties.json')
targetsPropertiesData = json.load(targetsPropertiesFile)

numImages = 240
receptive_size = (200, 200)
image_size = (1024, 1280)
screen_size = (1024, 1280)

trialsSequence = loadmat(trialsSequenceFile)
trialsSequence = trialsSequence['seq'].flatten()
trialsSequence = trialsSequence % numImages

subjectsFiles = listdir(subjectsFilesDir)
targets_found = 0
wrong_targets_found = 0

scanpaths_with_shorter_distance_than_receptive_size = 0
shortest_consecutive_fixations_coord_difference = (9999, 9999)
for subjectDataFile in subjectsFiles:
    if (not(subjectDataFile.endswith('_oracle.mat')) or not(subjectDataFile.endswith('.mat'))):
        continue
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("Processing " + subjectDataFile)

    subjectNumber = subjectDataFile[4:6]

    currentSubjectData = loadmat(subjectsFilesDir + subjectDataFile)
    subjectTrialsInfo = dict()
    stimuliProcessed = []
    for trialNumber in range(len(trialsSequence)):
        stimuli = trialsSequence[trialNumber]
        if (stimuli == 0): stimuli = 240
        # Only first trials are taken into account
        if (stimuli in stimuliProcessed):
            continue
        
        # Get fixation coordinates for the current trial; minus one, since Python indexes images from zero.
        fix_posX = currentSubjectData['FixData']['Fix_posx'][0][0][trialNumber][0].flatten() - 1
        fix_posY = currentSubjectData['FixData']['Fix_posy'][0][0][trialNumber][0].flatten() - 1

        number_of_fixations = len(fix_posX)
        target_found = False
        # TargetFound matrix has only 74 columns
        if number_of_fixations < 74:
            target_found = bool(currentSubjectData['FixData']['TargetFound'][0][0][trialNumber][(number_of_fixations - 1)])

        # Get target position information
        current_target = targetsPropertiesData[stimuli - 1]
        target_start_row = current_target['target_matched_row']
        target_start_column = current_target['target_matched_column']
        target_end_row = target_start_row + current_target['target_side_length']
        target_end_column = target_start_column + current_target['target_columns']

        target_bounding_box = [target_start_row, target_start_column, target_end_row, target_end_column]

        target_center = (target_start_row + (current_target['target_side_length'] // 2), target_start_column + (current_target['target_columns'] // 2))
        target_lower_row_bound  = target_center[0] - receptive_size[0]
        target_upper_row_bound  = target_center[0] + receptive_size[0]
        target_lower_column_bound = target_center[1] - receptive_size[1]
        target_upper_column_bound = target_center[1] + receptive_size[1]
        if (target_lower_row_bound < 0): target_lower_row_bound = 0
        if (target_upper_row_bound > image_size[0]): target_upper_row_bound = image_size[0]
        if (target_lower_column_bound < 0): target_lower_column_bound = 0
        if (target_upper_column_bound > image_size[1]): target_upper_column_bound = image_size[1]

        last_fixation_X = fix_posX[number_of_fixations - 1]
        last_fixation_Y = fix_posY[number_of_fixations - 1]
        between_bounds = (target_lower_row_bound <= last_fixation_Y) and (target_upper_row_bound >= last_fixation_Y) and (target_lower_column_bound <= last_fixation_X) and (target_upper_column_bound >= last_fixation_X)
        if target_found:
            if between_bounds:
                targets_found += 1
            else:
                print("Subject: " + subjectNumber + "; stimuli: " + str(stimuli) + "; trial: " + str(trialNumber) + ". Last fixation doesn't match target's bounds")
                print("Target's bounds: " + str((target_lower_row_bound, target_lower_column_bound, target_upper_row_bound, target_upper_column_bound)) + ". Last fixation: " + str((last_fixation_Y, last_fixation_X)) + '\n')
                wrong_targets_found += 1
                target_found = False
        
        if target_found and number_of_fixations > 1:
            fixations = [fix for fix in zip(fix_posX, fix_posY)]
            distance_between_consecutive_fixations = [np.linalg.norm(np.array(fix_1) - np.array(fix_2)) for fix_1, fix_2 in zip(fixations, fixations[1:])]
            shortest_distance_between_consecutive_fixations = min(distance_between_consecutive_fixations)
            fixation_number = distance_between_consecutive_fixations.index(shortest_distance_between_consecutive_fixations)
            coordinate_difference_of_shortest_consecutive_fixations = (abs(fix_posX[fixation_number] - fix_posX[fixation_number + 1]), abs(fix_posY[fixation_number] - fix_posY[fixation_number + 1]))
            if coordinate_difference_of_shortest_consecutive_fixations[0] < (receptive_size[0] / 2) and coordinate_difference_of_shortest_consecutive_fixations[1] < (receptive_size[1] / 2):
                scanpaths_with_shorter_distance_than_receptive_size += 1
                if coordinate_difference_of_shortest_consecutive_fixations[0] < shortest_consecutive_fixations_coord_difference[0] \
                    and coordinate_difference_of_shortest_consecutive_fixations[1] < shortest_consecutive_fixations_coord_difference[1]:
                    shortest_consecutive_fixations_coord_difference = coordinate_difference_of_shortest_consecutive_fixations
                    print("Subject: " + subjectNumber + "; stimuli: " + str(stimuli) + "; trial: " + str(trialNumber) + ". Fixation numbers: " + str(fixation_number + 1) + " " + str(fixation_number + 2) + ". Coord. difference: " + str(coordinate_difference_of_shortest_consecutive_fixations))

        fix_startTime = currentSubjectData['FixData']['Fix_starttime'][0][0][trialNumber][0].flatten()
        fix_endTime = currentSubjectData['FixData']['Fix_time'][0][0][trialNumber][0].flatten()
        fix_time = fix_endTime - fix_startTime

        stimuliName = str(stimuli)
        if stimuli < 10:
            stimuliName = '00' + stimuliName
        elif stimuli < 100:
            stimuliName = '0' + stimuliName
        imageName = 'img' + str(stimuliName) + '.jpg'

        subjectTrialsInfo[imageName] = { "subject" : subjectNumber, "dataset" : "IVSN Natural Design Dataset", "image_height" : image_size[0], "image_width" : image_size[1], "screen_height" : screen_size[0], "screen_width" : screen_size[1], "receptive_height" : receptive_size[0], "receptive_width" : receptive_size[1], \
            "target_found" : target_found, "target_bbox" : target_bounding_box, "X" : fix_posX.tolist(), "Y" : fix_posY.tolist(), "T" : fix_time.tolist(), "target_object" : "TBD", "max_fixations" : 80}
        stimuliProcessed.append(stimuli)
    subject_save_file = 'subj' + subjectNumber + '_scanpaths.json'
    if not(path.exists(save_path)):
        mkdir(save_path)
    with open(save_path + subject_save_file, 'w') as fp:
        json.dump(subjectTrialsInfo, fp, indent = 4)
        fp.close()

print("Targets found: " + str(targets_found) + ". Wrong targets found: " + str(wrong_targets_found))
print("Scanpaths with consecutive fixations in a shorter distance than 200x200: " + str(scanpaths_with_shorter_distance_than_receptive_size))
print("Min. coordinate difference in consecutive fixations:" + str(shortest_consecutive_fixations_coord_difference))
