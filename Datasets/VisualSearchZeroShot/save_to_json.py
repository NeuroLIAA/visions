from scipy.io import loadmat
import numpy as np
import json
from os import listdir

subjectsFilesDir = 'psy/ProcessScanpath_naturaldesign/'
trialsSequenceFile = 'psy/naturaldesign_seq.mat'
# Load targets locations
targetsPropertiesFile = open('../../Models/VisualSearchZeroShot/targets_locations.json')
targetsPropertiesData = json.load(targetsPropertiesFile)

numImages = 240
window_size = (200, 200)
image_size = (1024, 1280)

trialsSequence = loadmat(trialsSequenceFile)
trialsSequence = trialsSequence['seq'].flatten()
trialsSequence = trialsSequence % numImages

subjectsFiles = listdir(subjectsFilesDir)
subjectsTrialsInfo = []
targets_found = 0
wrong_targets_found = 0
for subjectDataFile in subjectsFiles:
    if (subjectDataFile.endswith('_oracle.mat') or not(subjectDataFile.endswith('.mat'))):
        continue

    subjectNumber = subjectDataFile[4:6]
    print("Processing " + subjectsFilesDir + subjectDataFile)
    currentSubjectData = loadmat(subjectsFilesDir + subjectDataFile)
    stimuliProcessed = []
    for trialNumber in range(len(trialsSequence)):
        stimuli = trialsSequence[trialNumber]
        if (stimuli == 0): stimuli = 240
        if (stimuli in stimuliProcessed):
            continue
        
        # Get fixation coordinates for the current trial; minus one, since Python indexes images from zero.
        fix_posX = currentSubjectData['FixData']['Fix_posx'][0][0][trialNumber][0].flatten() - 1
        fix_posY = currentSubjectData['FixData']['Fix_posy'][0][0][trialNumber][0].flatten() - 1
        # First fixation is always at the center of the stimuli, it isn't taken into account in the number of fixations
        number_of_fixations = len(fix_posX) - 1
        target_found = False
        # TargetFound matrix only has 74 columns
        if (number_of_fixations < 74):
            target_found = bool(currentSubjectData['FixData']['TargetFound'][0][0][trialNumber][(number_of_fixations - 1)])
            target_found = target_found or bool(currentSubjectData['FixData']['TargetFound'][0][0][trialNumber][(number_of_fixations)])

        # Get target position information
        current_target = targetsPropertiesData[stimuli - 1]
        target_start_row = current_target['matched_row']
        target_start_column = current_target['matched_column']
        target_end_row = target_start_row + current_target['target_side_length']
        target_end_column = target_start_column + current_target['target_columns']

        target_bounding_box = [target_start_row, target_start_column, target_end_row, target_end_column]

        target_center = (target_start_row + (current_target['target_side_length'] // 2), target_start_column + (current_target['target_columns'] // 2))
        target_lower_row_bound  = target_center[0] - window_size[0]
        target_upper_row_bound  = target_center[0] + window_size[0]
        target_lower_column_bound = target_center[1] - window_size[1]
        target_upper_column_bound = target_center[1] + window_size[1]
        if (target_lower_row_bound < 0): target_lower_row_bound = 0
        if (target_upper_row_bound > image_size[0]): target_upper_row_bound = image_size[0]
        if (target_lower_column_bound < 0): target_lower_column_bound = 0
        if (target_upper_column_bound > image_size[1]): target_upper_column_bound = image_size[1]

        last_fixation_X = fix_posX[number_of_fixations]
        last_fixation_Y = fix_posY[number_of_fixations]
        between_bounds = (target_lower_row_bound <= last_fixation_Y) and (target_upper_row_bound >= last_fixation_Y) and (target_lower_column_bound <= last_fixation_X) and (target_upper_column_bound >= last_fixation_X)
        if (target_found):
            if (between_bounds):
                targets_found += 1
            else:
                print("Subject: " + subjectNumber + "; stimuli: " + str(stimuli) + "; trial: " + str(trialNumber) + ". Last fixation doesn't match target's bounds")
                print("Target's bounds: " + str((target_lower_row_bound, target_lower_column_bound, target_upper_row_bound, target_upper_column_bound)) + ". Last fixation: " + str((last_fixation_Y, last_fixation_X)))
                wrong_targets_found += 1

        fix_startTime = currentSubjectData['FixData']['Fix_starttime'][0][0][trialNumber][0].flatten()
        fix_endTime = currentSubjectData['FixData']['Fix_time'][0][0][trialNumber][0].flatten()
        fix_time = fix_endTime - fix_startTime

        stimuliName = str(stimuli)
        if (stimuli < 10):
            stimuliName = '00' + stimuliName
        elif (stimuli < 100):
            stimuliName = '0' + stimuliName
        imageName = 'img' + str(stimuliName) + '.jpg'

        subjectsTrialsInfo.append({ "subject" : subjectNumber, "image" : imageName, "dataset" : "VisualSearchZeroShot Natural Design Dataset", "image_height" : image_size[0], "image_width" : image_size[1], "window_height" : window_size[0], "window_width" : window_size[1], \
            "target_found" : str(target_found), "target_bbox" : target_bounding_box, "X" : fix_posX.tolist(), "Y" : fix_posY.tolist(), "T" : fix_time.tolist(), "split" : "valid", "target_object" : "te la debo", "max_fixations" : "80"})
        stimuliProcessed.append(stimuli)

print("Targets found: " + str(targets_found) + ". Wrong targets found: " + str(wrong_targets_found))
jsonStructsFile = open(subjectsFilesDir + 'human_scanpaths.json', 'w')
json.dump(subjectsTrialsInfo, jsonStructsFile, indent = 4)
jsonStructsFile.close()
