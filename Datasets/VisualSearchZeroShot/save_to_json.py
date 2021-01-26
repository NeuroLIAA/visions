from scipy.io import loadmat
import numpy as np
import json
from os import listdir

def rescaleCoordinates(start_row, start_column, end_row, end_column, img_height, img_width, new_img_height, new_img_width):
    rescaled_start_row = round((start_row / img_height) * new_img_height)
    rescaled_start_column = round((start_column / img_width) * new_img_width)
    rescaled_end_row = round((end_row / img_height) * new_img_height)
    rescaled_end_column = round((end_column / img_width) * new_img_width)

    return rescaled_start_row, rescaled_start_column, rescaled_end_row, rescaled_end_column

subjectsFilesDir = 'psy/ProcessScanpath_naturaldesign/'
trialsSequenceFile = 'psy/naturaldesign_seq.mat'
# Load targets positions
targetsPropertiesFile = open('../../Models/VisualSearchZeroShot/targets_positions.json')
targetsPropertiesData = json.load(targetsPropertiesFile)

numImages = 240

trialsSequence = loadmat(trialsSequenceFile)
trialsSequence = trialsSequence['seq'].flatten()
trialsSequence = trialsSequence % numImages

subjectsFiles = listdir(subjectsFilesDir)
subjectsTrialsInfo = []
for subjectDataFile in subjectsFiles:
    if (subjectDataFile.endswith('_oracle.mat') or not(subjectDataFile.endswith('.mat'))):
        continue

    subjectNumber = subjectDataFile[4:6]
    print("Processing " + subjectsFilesDir + subjectDataFile)
    currentSubjectData = loadmat(subjectsFilesDir + subjectDataFile)
    stimuliProcessed = np.empty(numImages * 17)
    for trialNumber in range(len(trialsSequence)):
        stimuli = trialsSequence[trialNumber]
        if (stimuli == 0): stimuli = 240
        if (stimuli in stimuliProcessed):
            continue
        
        # Get fixation coordinates for the current trial
        fix_posX = currentSubjectData['FixData']['Fix_posx'][0][0][trialNumber][0].flatten()
        fix_posY = currentSubjectData['FixData']['Fix_posy'][0][0][trialNumber][0].flatten()
        # First fixation is always at the center of the stimuli, it isn't taken into account in the number of fixations
        number_of_fixations = len(fix_posX) - 1
        target_found = False
        # TargetFound matrix only has 74 columns
        if (number_of_fixations < 74):
            target_found = bool(currentSubjectData['FixData']['TargetFound'][0][0][trialNumber][(number_of_fixations - 1)])

        # Get target position information
        current_target = targetsPropertiesData[stimuli - 1]
        target_start_row = current_target['matched_row']
        target_start_column = current_target['matched_column']
        target_end_row = target_start_row + current_target['target_side_length']
        target_end_column = target_start_column + current_target['target_columns']
        # Image size was different, rescale positions
        rescaled_tg_start_row, rescaled_tg_start_column, rescaled_tg_end_row, rescaled_tg_end_column = \
             rescaleCoordinates(target_start_row, target_start_column, target_end_row, target_end_column, current_target['image_height'], current_target['image_width'], 1028, 1280) 
            
        target_bounding_box = [rescaled_tg_start_row, rescaled_tg_start_column, rescaled_tg_end_row, rescaled_tg_end_column]

        last_fixation_X = fix_posX[len(fix_posX) - 1]
        last_fixation_Y = fix_posY[len(fix_posY) - 1]
        between_bounds = ((rescaled_tg_start_row - 10 < last_fixation_Y) and (rescaled_tg_end_row + 10 > last_fixation_Y)) and ((rescaled_tg_start_column - 10 < last_fixation_X) and (rescaled_tg_end_column + 10 > last_fixation_X))
        if (target_found and not(between_bounds)):
            print("Subject: " + subjectNumber + ", stimuli: " + str(stimuli) + ". Last fixation doesn't match target bounds")

        fix_startTime = currentSubjectData['FixData']['Fix_starttime'][0][0][trialNumber][0].flatten()
        fix_endTime = currentSubjectData['FixData']['Fix_time'][0][0][trialNumber][0].flatten()
        fix_time = fix_endTime - fix_startTime

        stimuliName = str(stimuli)
        if (stimuli < 10):
            stimuliName = '00' + stimuliName
        elif (stimuli < 100):
            stimuliName = '0' + stimuliName
        imageName = 'img' + str(stimuliName) + '.jpg'

        subjectsTrialsInfo.append({ "subject" : subjectNumber, "image" : imageName, "image_height" : 1028, "image_width" : 1280, "X" : fix_posX.tolist(), "Y" : fix_posY.tolist(), "T" : fix_time.tolist(), "target_bbox" : target_bounding_box, "dataset" : "VisualSearchZeroShot Natural Design Dataset", \
            "split" : "valid", "target object" : "te la debo", "maximum fixations" : "80", "target found"  : str(target_found) })
        np.append(stimuliProcessed, stimuli)

jsonStructsFile = open(subjectsFilesDir + 'human_scanpaths.json', 'w')
json.dump(subjectsTrialsInfo, jsonStructsFile, indent = 4)
jsonStructsFile.close()
