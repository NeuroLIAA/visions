from os import mkdir, listdir, path
from skimage import io, color, transform, img_as_ubyte

def main():
    # Preprocessing of images (conversion to grayscale, resizing and dividing into blocks)
    preprocess_images()

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
    # Create blocks of size 224x224, if possible
    row = 0
    column = 0
    for block_number in range(30):
        if row < 4:
            if column < 5:
                block_size = (224, 224)
            else:
                block_size = (224, 160)
        elif column < 5:
            block_size = (132, 224)
        else:
            block_size = (132, 160)
            
        from_x = column * 224
        from_y = row * 224
        to_y = from_y + block_size[0]
        to_x = from_x + block_size[1]
        img_crop = image[from_y:to_y, from_x:to_x]
        io.imsave(save_path + '/img_id' + imgID + '_' + str(block_number) + '.jpg', img_as_ubyte(img_crop), check_contrast=False)
        column += 1
        if column == 6:
            column = 0
            row += 1

main()