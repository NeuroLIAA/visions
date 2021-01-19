import subprocess
import scipy.io
import numpy as np
import json
from os import mkdir, listdir, path
from skimage import io, color, transform, img_as_ubyte, exposure


def main():
    stimuliDir = 'stimuli/'
    choppedDir = 'choppednaturaldesign/'
    stimuliSize = (1028, 1280)

    preprocess_images(stimuliDir, choppedDir, stimuliSize)
    run_model()
    compute_scanpaths(stimuliDir, choppedDir, stimuliSize)

def compute_scanpaths(stimuliDir, choppedDir, stimuliSize):
    enumeratedImages = listdir(stimuliDir)
    layerList = np.array([1])

    for imageName in enumeratedImages:
        if not(imageName.endswith('.jpg')):
            continue
        
        imgID = imageName[3:-4]
        choppedImgDir = choppedDir + 'img' + imgID
        choppedData = listdir(choppedImgDir)
        template    = np.zeros([stimuliSize[0], stimuliSize[1]])
        for layer in range(len(layerList)):
            for choppedSaliencyData in choppedData:
                if (choppedSaliencyData.endswith('.jpg')):
                    continue

                imgName    = choppedSaliencyData[:-17]
                choppedImg = io.imread(choppedImgDir + '/' + imgName + '.jpg')
                # Load data computed by the model
                choppedSaliencyImg = scipy.io.loadmat(choppedImgDir + '/' + choppedSaliencyData)
                choppedSaliencyImg = choppedSaliencyImg['x']
                choppedSaliencyImg = transform.resize(choppedSaliencyImg, choppedImg.shape)
                # Get coordinate information from the chopped image name
                imgNameSplit = imgName.split('_')
                from_row     = int(imgNameSplit[2]) - 1
                from_column  = int(imgNameSplit[3]) - 1
                # Replace in template
                template[from_row:(from_row + choppedImg.shape[0]), from_column:(from_column + choppedImg.shape[1])] = choppedSaliencyImg
            saliencyImg = img_as_ubyte(exposure.rescale_intensity(template))
            # For debugging
            io.imsave(imgID + '_saliency.jpg', saliencyImg)

            # Compute scanpaths from saliency image
            xCoordFixationOrder = []
            yCoordFixationOrder = []
            # json encoding
            prejsonStruct = { "X" : xCoordFixationOrder , "Y" : yCoordFixationOrder, "dataset" : "VisualSearchZeroShot Natural Design Dataset", "image" : imgID + ".jpg", "split" : "test", "subject" : "VisualSearchZeroShot Model" , "target" : "te la debo" }
            prejsonstructs = prejsonstructs.append(prejsonStruct)
    


    jsonStructs = json.dumps(prejsonstructs,indent = 4)


    jsonStructsFile = open("scanpathspython.json","w")
    jsonStructsFile.write(jsonStructs)
    jsonStructsFile.close();


def run_model():
    subprocess.run("th IVSNtopdown_30_31_naturaldesign.lua", shell=True, check=True)

def preprocess_images(stimuliDir, choppedDir, stimuliSize):
    # Preprocessing of images (conversion to grayscale, resizing and dividing into blocks)
    enumeratedImages = listdir(stimuliDir)

    for imageName in enumeratedImages:
        if not(imageName.endswith('.jpg')):
            continue

        img = io.imread(stimuliDir + imageName)
        if len(img.shape) >= 3:
            img = color.rgb2gray(img)
        img = transform.resize(img, stimuliSize)

        imgID = imageName[3:-4]
        choppedImgDir = choppedDir + 'img' + imgID
        if not(path.exists(choppedImgDir)):
            mkdir(choppedImgDir)

        divide_into_blocks(img, imgID, choppedImgDir)

def divide_into_blocks(image, imgID, save_path):
    # Create blocks of size 224 x 224
    img_height, img_width = image.shape[0], image.shape[1]
    block_height, block_width = 224, 224

    number_of_rows    = img_height // block_height
    number_of_columns = img_width // block_width
    extra_row    = (img_height % block_height > 0)
    extra_column = (img_width % block_width > 0)
    if (extra_row):
        extra_row_height = img_height % block_height
        number_of_rows += 1
    if (extra_column):
        extra_column_width = img_width % block_width
        number_of_columns += 1
    
    for row in range(number_of_rows):
        block_size = (block_height, block_width)
        if (extra_row and (row + 1) == number_of_rows):
            block_size = (extra_row_height, block_size[1])
        for column in range(number_of_columns):
            if (extra_column and (column + 1) == number_of_columns):
                block_size = (block_size[0], extra_column_width)
            from_x = column * block_width
            from_y = row * block_height
            to_y = from_y + block_size[0]
            to_x = from_x + block_size[1]
            img_crop = image[from_y:to_y, from_x:to_x]
            io.imsave(save_path + '/img_id' + imgID + '_' + str(from_y + 1) + '_' + str(from_x + 1) + '.jpg', img_as_ubyte(img_crop), check_contrast=False)

main()
