import json
import shutil
from os import listdir, path, mkdir
from skimage import io, transform
from utils import rescale_coordinate, rename_image

""" This script requires that all original images of the MCS dataset are placed in the '../images' folder """

test_images_dir = '../images/test/TP/'
train_microwave_images_dir = '../images/train/microwave/TP/'
train_clock_images_dir     = '../images/train/clock/TP/'

targets_bboxes_file = 'targets_bboxes.json'
with open(targets_bboxes_file, 'r') as fp:
    targets_bboxes = json.load(fp)
# Average size of all images
new_image_size = (508, 564)
initial_fixation = (new_image_size[0] / 2, new_image_size[1] / 2)

trials_properties = []
trials_properties_file = '../trials_properties.json'
targets_save_path   = '../targets/'
images_save_path    = '../images/'
categories_path     = 'categories'

test_images = listdir(test_images_dir)
train_microwave_images = listdir(train_microwave_images_dir)
train_clock_images = listdir(train_clock_images_dir)

number_of_test_images  = len(test_images)
number_of_clock_images = len(train_clock_images)
images_files = test_images + train_clock_images + train_microwave_images
for index in range(len(images_files)):
    if index < number_of_test_images:
        image_path = test_images_dir
        image_categories = ['microwave', 'clock']
    elif index < number_of_test_images + number_of_clock_images:
        image_path = train_clock_images_dir
        image_categories = ['clock']
    else:
        image_path = train_microwave_images_dir
        image_categories = ['microwave']
    
    image_name = images_files[index]
    image = io.imread(path.join(image_path, image_name))
    image = transform.resize(image, new_image_size)

    for category in image_categories:
        new_image_name = rename_image(image_name, category)
        if not new_image_name[1:] in targets_bboxes[category]:
            print('Image not found in targets bounding boxes: ' + image_name + ', task: ' + category)
            continue

        img_info = targets_bboxes[category][new_image_name[1:]]
        original_img_size = (img_info['image_height'], img_info['image_width'])
        target_bbox = img_info['target_bbox']

        rescaled_target_bbox = [int(rescale_coordinate(target_bbox[i], original_img_size[i % 2 == 1], new_image_size[i % 2 == 1])) for i in range(len(target_bbox))]
        # Edge cases
        rescaled_target_bbox[2] = min(new_image_size[0] - 1, rescaled_target_bbox[2])
        rescaled_target_bbox[3] = min(new_image_size[1] - 1, rescaled_target_bbox[3])
        
        target_height = rescaled_target_bbox[2] - rescaled_target_bbox[0]
        target_width  = rescaled_target_bbox[3] - rescaled_target_bbox[1]
        
        target_name = new_image_name[:-4] + '_target' + new_image_name[-4:]
        if not path.exists(targets_save_path):
            mkdir(targets_save_path)
        shutil.copyfile(path.join(categories_path, category + '.jpg'), path.join(targets_save_path, target_name))

        trials_properties.append({'image' : new_image_name, 'target' : target_name, 'dataset' : 'MCS Dataset', \
            'target_matched_row' : rescaled_target_bbox[0], 'target_matched_column' : rescaled_target_bbox[1], 'target_height' : target_height, 'target_width' : target_width, \
                'image_height' : new_image_size[0], 'image_width' : new_image_size[1], 'initial_fixation_row' : int(initial_fixation[0]), 'initial_fixation_column' : int(initial_fixation[1]), \
                    'target_object' : category})

with open(trials_properties_file, 'w') as fp:
    json.dump(trials_properties, fp, indent=4)