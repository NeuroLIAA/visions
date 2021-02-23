from skimage import io, color, transform, img_as_ubyte
from os import listdir, path, mkdir

"""
Preprocessing of images (resizing and dividing into blocks)
Stimuli is divided into blocks of size 224x224, which are then fed to the CNN.
Files are saved in chopped_dir.
"""

def chop_stimuli(stimuli_dir, chopped_dir, stimuli_size, targets_locations):
    for struct in targets_locations:
        image_name = struct['image']

        imgID = image_name[:-4]
        img = io.imread(stimuli_dir + image_name)
        img = transform.resize(img, stimuli_size)

        if not(path.exists(chopped_dir)):
            mkdir(chopped_dir)

        choppedImgDir = chopped_dir + imgID
        if not(path.exists(choppedImgDir)):
            mkdir(choppedImgDir)

        divide_into_blocks(img, imgID, choppedImgDir)

def divide_into_blocks(image, imgID, save_path):
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
            io.imsave(save_path + '/' + imgID + '_' + str(from_row) + '_' + str(from_column) + '.jpg', img_as_ubyte(img_crop), check_contrast=False)
