import subprocess
import scipy.io
import numpy as np
import pandas as pd
import json
import cv2 as ocv
from os import mkdir, listdir, path
from skimage import io, color, transform, img_as_ubyte, exposure

stimuliDir = 'stimuli/'
choppedDir = 'choppednaturaldesign/'
resultsDir = 'results/'
stimuliSize = (1028, 1280)

def main():
    preprocess_images()
    run_model()
    compute_scanpaths()

def compute_scanpaths():
    enumeratedImages = listdir(stimuliDir)
    scanpaths = []

    for imageName in enumeratedImages:
        if not(imageName.endswith('.jpg')):
            continue
        
        saliencyImg = load_model_data(imageName)
        #io.imsave("saliency004_python.jpg", saliencyImg)

        maxFixations  = 80
        receptiveSize = 200
        # Target coordinates are True values in target_location
        target_location = io.imread(stimuliDir + 'gt/' + imageName)
        target_location = color.rgb2gray(transform.resize(target_location, stimuliSize))
        target_location = target_location > 0.5

        xCoordFixationOrder = []
        yCoordFixationOrder = []
        target_found = False
        # Compute scanpaths from saliency image        
        for fixationNumber in range(maxFixations):
            coordinates = np.where(saliencyImg == np.amax(saliencyImg))
            posX = coordinates[0][0]
            posY = coordinates[1][0]

            xCoordFixationOrder.append(str(posX))
            yCoordFixationOrder.append(str(posY))

            fixatedPlace_leftX  = posX - receptiveSize // 2 + 1
            fixatedPlace_rightX = posX + receptiveSize // 2
            fixatedPlace_leftY  = posY - receptiveSize // 2 + 1
            fixatedPlace_rightY = posY + receptiveSize // 2

            if fixatedPlace_leftX < 0: fixatedPlace_leftX = 0
            if fixatedPlace_leftY < 0: fixatedPlace_leftY = 0
            if fixatedPlace_rightX > stimuliSize[0]: fixatedPlace_rightX = stimuliSize[0]
            if fixatedPlace_rightY > stimuliSize[1]: fixatedPlace_rightY = stimuliSize[1]

            fixatedPlace = target_location[fixatedPlace_leftX:fixatedPlace_rightX, fixatedPlace_leftY:fixatedPlace_rightY]

            if (np.sum(fixatedPlace) > 0):
                target_found = True
                break
            else:
                # Apply inhibition of return
                saliencyImg[fixatedPlace_leftX:fixatedPlace_rightX, fixatedPlace_leftY:fixatedPlace_rightY] = 0
                
        if (target_found):
            print(imageName + "; target found at fixation step " + str(fixationNumber + 1))
        # JSON encoding
        scanpaths.append({ "image" : imageName, "dataset" : "VisualSearchZeroShot Natural Design Dataset", "subject" : "VisualSearchZeroShot Model", "target_found"  : str(target_found), "X" : xCoordFixationOrder, "Y" : yCoordFixationOrder,  "split" : "test", \
            "image_height" : stimuliSize[0], "image_width" : stimuliSize[1], "target_object" : "te la debo" , "max_fixations" : str(maxFixations)})
    
    jsonStructsFile = open(resultsDir + 'scanpathspython.json', 'w')
    json.dump(scanpaths, jsonStructsFile, indent = 4)
    jsonStructsFile.close()

def load_model_data(imageName):
    # Get saliency map for stimuli
    imgID = imageName[3:-4]
    choppedImgDir = choppedDir + 'img' + imgID + '/'
    choppedData = listdir(choppedImgDir)

    template  = np.zeros([stimuliSize[0], stimuliSize[1]])
    layerList = np.array([1])
    for layer in range(len(layerList)):
        for choppedSaliencyData in choppedData:
            if (choppedSaliencyData.endswith('.jpg')):
                continue
            choppedImgName = choppedSaliencyData[:-17]
            choppedImg     = io.imread(choppedImgDir + choppedImgName + '.jpg')
            choppedImg_height = choppedImg.shape[0]
            choppedImg_width  = choppedImg.shape[1]
            # Load data computed by the model
            choppedSaliencyImg = scipy.io.loadmat(choppedImgDir + choppedSaliencyData)
            choppedSaliencyImg = choppedSaliencyImg['x']
            #choppedSaliencyImg = transform.resize(choppedSaliencyImg, (choppedImg_height, choppedImg_width), order=3, anti_aliasing=True)
            choppedSaliencyImg = ocv.resize(choppedSaliencyImg, (choppedImg_width, choppedImg_height), interpolation=ocv.INTER_CUBIC)
            # Get coordinate information from the chopped image name
            choppedImgNameSplit = choppedImgName.split('_')
            from_row    = int(choppedImgNameSplit[2])
            from_column = int(choppedImgNameSplit[3])
            to_row    = from_row + choppedImg_height
            to_column = from_column + choppedImg_width
            # Replace in template
            template[from_row:to_row, from_column:to_column] = choppedSaliencyImg
        saliencyImg = exposure.rescale_intensity(template, out_range=(0, 1))
        #df = pd.DataFrame(saliencyImg)
        #df.to_csv("saliency_python.csv", index=False)
    
    return saliencyImg

def run_model():
    subprocess.run("th IVSNtopdown_30_31_naturaldesign.lua", shell=True, check=True)

def preprocess_images():
    # Preprocessing of images (conversion to grayscale, resizing and dividing into blocks)
    enumeratedImages = listdir(stimuliDir)

    for imageName in enumeratedImages:
        if not(imageName.endswith('.jpg')):
            continue

        imgID = imageName[3:-4]
        img = io.imread(stimuliDir + imageName)
        if len(img.shape) >= 3:
            img = color.rgb2gray(img)
        img = transform.resize(img, stimuliSize)

        if not(path.exists(choppedDir)):
            mkdir(choppedDir)

        choppedImgDir = choppedDir + 'img' + imgID
        if not(path.exists(choppedImgDir)):
            mkdir(choppedImgDir)

        divide_into_blocks(img, imgID, choppedImgDir)

def divide_into_blocks(image, imgID, save_path):
    # Create blocks of size 224 x 224
    img_height, img_width = image.shape[0], image.shape[1]
    default_block_height, default_block_width = 224, 224

    number_of_rows    = img_height // default_block_height
    number_of_columns = img_width // default_block_width
    extra_row    = (img_height % default_block_height > 0)
    extra_column = (img_width % default_block_width > 0)
    if (extra_row):
        extra_row_height = img_height % default_block_height
        number_of_rows += 1
    if (extra_column):
        extra_column_width = img_width % default_block_width
        number_of_columns += 1
    
    for row in range(number_of_rows):
        current_block_size = (default_block_height, default_block_width)
        if (extra_row and (row + 1) == number_of_rows):
            current_block_size = (extra_row_height, current_block_size[1])
        for column in range(number_of_columns):
            if (extra_column and (column + 1) == number_of_columns):
                current_block_size = (current_block_size[0], extra_column_width)
            
            from_row    = row * default_block_height
            from_column = column * default_block_width
            to_row    = from_row + current_block_size[0]
            to_column = from_column + current_block_size[1]
            img_crop = image[from_row:to_row, from_column:to_column]
            io.imsave(save_path + '/img_id' + imgID + '_' + str(from_row) + '_' + str(from_column) + '.jpg', img_as_ubyte(img_crop), check_contrast=False)

main()
