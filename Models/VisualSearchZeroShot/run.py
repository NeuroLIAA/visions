import subprocess
from os import mkdir, listdir, path
from skimage import io, color, transform, img_as_ubyte

def main():
    # Preprocessing of images (conversion to grayscale, resizing and dividing into blocks)
    preprocess_images()
    
    run_model()

def run_model():
    subprocess.run("th IVSNtopdown_30_31_naturaldesign.lua", shell=True, check=True)

def preprocess_images():
    stimuliFolder = 'stimuli/'
    enumeratedImages = listdir(stimuliFolder)

    for imageName in enumeratedImages:
        if not(imageName.endswith('.jpg')):
            continue

        img = io.imread(stimuliFolder + imageName)
        if len(img.shape) >= 3:
            img = color.rgb2gray(img)
        img = transform.resize(img, (1028, 1280))

        imgID = imageName[3:-4]
        choppedDir = 'choppednaturaldesign/img' + imgID
        if not(path.exists(choppedDir)):
            mkdir(choppedDir)

        divide_into_blocks(img, imgID, choppedDir)

def divide_into_blocks(image, imgID, save_path):
    # Create blocks of size 224 x 224
    img_height, img_width = image.shape[0], image.shape[1]
    block_height, block_width = 224, 224

    amount_of_rows    = img_height // block_height
    amount_of_columns = img_width // block_width
    extra_row    = (img_height % block_height > 0)
    extra_column = (img_width % block_width > 0)
    if (extra_row):
        extra_row_height = img_height % block_height
        amount_of_rows += 1
    if (extra_column):
        extra_column_width = img_width % block_width
        amount_of_columns += 1
    
    block_number = 1
    for row in range(amount_of_rows):
        block_size = (block_height, block_width)
        if (extra_row and (row + 1) == amount_of_rows):
            block_size = (extra_row_height, block_size[1])
        for column in range(amount_of_columns):
            if (extra_column and (column + 1) == amount_of_columns):
                block_size = (block_size[0], extra_column_width)
            from_x = column * block_width
            from_y = row * block_height
            to_y = from_y + block_size[0]
            to_x = from_x + block_size[1]
            img_crop = image[from_y:to_y, from_x:to_x]
            io.imsave(save_path + '/img_id' + imgID + '_' + str(block_number) + '.jpg', img_as_ubyte(img_crop), check_contrast=False)
            block_number += 1


main()