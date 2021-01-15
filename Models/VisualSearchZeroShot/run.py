import os
from skimage import io, color, transform


# Preprocessing of images (conversion to grayscale, resizing and cropping)
stimuliFolder = 'stimuli/'
enumeratedImages = os.listdir(stimuliFolder)

for imageName in enumeratedImages:
    if !(imageName.endswith('.jpg')):
        continue

    img = io.imread(stimuliFolder + imageName)
    if len(img.shape) >= 3:
        img = color.rgb2gray(img)
    img = transform.resize(img, (1028, 1280), anti_aliasing=True)

    imgID = imageName[3:-4]
    choppedDir = 'choppednaturaldesign/img' + imgID
    if !(os.path.exists(choppedDir)):
        os.mkdir(choppedDir)

    #### TODO: Crop in blocks of 224 x 224 ####