from os import listdir
from skimage import io, color

stimuliDir = '../stimuli/'
targetDir = '../target/'

stimuliFiles = sorted(listdir(stimuliDir))
targetFiles = sorted(listdir(targetDir))

for stimuliFile in stimuliFiles:
    if not(stimuliFile.endswith('.jpg')):
        continue

    stimuliImg = io.imread(stimuliDir + stimuliFile)
    stimuliImg = color.rgb2gray(stimuliImg)
    io.imsave(stimuliDir + stimuliFile, stimuliImg)

for targetFile in targetFiles:
    if not(targetFile.endswith('.jpg')):
        continue

    targetImg = io.imread(targetDir + targetFile)
    targetImg = color.rgb2gray(targetImg)
    io.imsave(targetDir + targetFile, targetImg)