from scipy.io import loadmat
import pandas as pd
import numpy as np
import json
from os import listdir

subjectsFilesDir = 'psy/ProcessScanpath_naturaldesign/'
trialsSequenceFile = 'psy/naturaldesign_seq.mat'
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
        if (stimuli in stimuliProcessed):
            continue
        
        # Get fixation coordinates for the current trial
        fix_posX = currentSubjectData['FixData']['Fix_posx'][0][0][trialNumber][0].flatten()
        fix_posY = currentSubjectData['FixData']['Fix_posy'][0][0][trialNumber][0].flatten()
        # First fixation is always at the center of the stimuli, it isn't taken into account in the number of fixations
        number_of_fixations = len(fix_posX) - 1
        target_found = False
        # Target found matrix has only 74 columns
        if (number_of_fixations < 74):
            target_found = bool(currentSubjectData['FixData']['TargetFound'][0][0][trialNumber][(number_of_fixations - 1)])

        fix_startTime = currentSubjectData['FixData']['Fix_starttime'][0][0][trialNumber][0].flatten()
        fix_endTime = currentSubjectData['FixData']['Fix_time'][0][0][trialNumber][0].flatten()
        fix_time = fix_endTime - fix_startTime

        if (stimuli == 0): stimuli = 240
        stimuliName = str(stimuli)
        if (stimuli < 10):
            stimuliName = '00' + stimuliName
        elif (stimuli < 100):
            stimuliName = '0' + stimuliName
        imageName = 'img' + str(stimuliName) + '.jpg'

        subjectsTrialsInfo.append({ "X" : fix_posX.tolist(), "Y" : fix_posY.tolist(), "T" : fix_time.tolist(), "dataset" : "VisualSearchZeroShot Natural Design Dataset", "image" : imageName, "split" : "valid", "subject" : subjectNumber, "target object" : "te la debo", "maximum fixations" : "80", "target found"  : str(target_found) })
        np.append(stimuliProcessed, [stimuli])

jsonStructsFile = open(subjectsFilesDir + 'human_scanpaths.json', 'w')
json.dump(subjectsTrialsInfo, jsonStructsFile, indent = 4)
jsonStructsFile.close()
