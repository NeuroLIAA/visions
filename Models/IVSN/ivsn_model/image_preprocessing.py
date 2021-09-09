from skimage import io, transform, img_as_ubyte
from os import listdir, path, mkdir, makedirs

"""
Preprocessing of images (resizing and dividing into blocks)
Images are divided into blocks of size 224x224, which are then fed to the CNN.
Block files are saved in preprocessed_images_dir, inside the corresponding dataset and image folder.
"""

def chop_images(stimuli_dir, preprocessed_images_dir, stimuli_size, trials_properties):
    for trial_properties in trials_properties:
        image_name = trial_properties['image']

        img = io.imread(stimuli_dir + image_name)
        if (img.shape[0], img.shape[1]) != stimuli_size:
            img = transform.resize(img, stimuli_size)

        if not(path.exists(preprocessed_images_dir)):
            makedirs(preprocessed_images_dir)

        img_id = image_name[:-4]
        chopped_img_dir = preprocessed_images_dir + img_id
        if not(path.exists(chopped_img_dir)):
            mkdir(chopped_img_dir)

        divide_into_blocks(img, img_id, chopped_img_dir)

def divide_into_blocks(image, img_id, save_path):
    img_height, img_width = image.shape[0], image.shape[1]
    default_block_height, default_block_width = 224, 224

    number_of_rows    = img_height // default_block_height
    number_of_columns = img_width // default_block_width
    extra_row    = (img_height % default_block_height > 0)
    extra_column = (img_width % default_block_width > 0)
    if extra_row:
        extra_row_height = img_height % default_block_height
        number_of_rows += 1
    if extra_column:
        extra_column_width = img_width % default_block_width
        number_of_columns += 1
    
    for row in range(number_of_rows):
        current_block_size = (default_block_height, default_block_width)
        if extra_row and (row + 1) == number_of_rows:
            current_block_size = (extra_row_height, current_block_size[1])
        for column in range(number_of_columns):            
            if extra_column and (column + 1) == number_of_columns:
                current_block_size = (current_block_size[0], extra_column_width)
            from_row    = row * default_block_height
            from_column = column * default_block_width
            to_row    = from_row + current_block_size[0]
            to_column = from_column + current_block_size[1]
            img_crop = image[from_row:to_row, from_column:to_column]
            
            block_filename = save_path + '/' + img_id + '_' + str(from_row) + '_' + str(from_column) + '.jpg'
            if not(path.exists(block_filename)):
                io.imsave(block_filename, img_as_ubyte(img_crop), check_contrast=False)
