import tensorflow_datasets as tfds
import numpy as np
import json
from os import listdir, path

""" Gets the targets bounding box from the coco2014 dataset """

ds, info = tfds.load('coco', split=['train', 'validation'], with_info=True, 
    shuffle_files=False, data_dir='/media/fert/TOURO Mobile/tensorflow_datasets')

microwave_id = 68
clock_id = 74

test_images_path  = './images/test/TP/'
train_images_path = './images/train/'
class_labels = [{'name': 'microwave', 'id': microwave_id}, {'name': 'clock', 'id': clock_id}]

save_file = 'targets_bboxes.json'

# Test images have both categories in every image
test_images_names  = listdir(test_images_path)

train_images_names = []
train_images_categories = []
for category in class_labels:
    name = category['name']
    category_images_path = path.join(path.join(train_images_path, name), 'TP')
    images_names = listdir(category_images_path)
    
    train_images_names += images_names
    train_images_categories += [category['id']] * len(images_names)

all_images = train_images_names + test_images_names
# Add zeros to the names, so they can be located in the dataset
for index in range(len(all_images)):
    new_name = all_images[index]
    for zeros in range(12 - len(new_name[:-4])):
        new_name = '0' + new_name
    all_images[index] = new_name

last_train_image = len(train_images_names)
targets_bboxes   = {'microwave': {}, 'clock': {}}
remaining_images = list(all_images)

for dts in ds:
    for instance in dts:
        if len(remaining_images) == 0:
            break
        name = str(instance['image/filename'].numpy())[17:][:-1]
        if name in remaining_images:
            index_in_list = all_images.index(name)
            if index_in_list >= last_train_image:
                categories = [microwave_id, clock_id]
            else:
                categories = [train_images_categories[index_in_list]]

            img_height, img_width = instance['image'].numpy().shape[:2]
            bboxes = instance['objects']['bbox'].numpy()
            labels = instance['objects']['label'].numpy()

            for category in categories:
                bbox_index  = np.where(category == labels)[0][0]
                target_bbox = bboxes[bbox_index] * [img_height, img_width, img_height, img_width]
                if category == microwave_id:
                    category_name = 'microwave'
                else:
                    category_name = 'clock'
                
                targets_bboxes[category_name][name] = {'image_height': img_height, 'image_width': img_width, 'target_bbox': list(target_bbox)}
            
            remaining_images.remove(name)

if len(remaining_images) > 0:
    print('There are images on which the bounding box coudln\'t be retrieved!')
    print(remaining_images)

with open(save_file, 'w') as fp:
    json.dump(targets_bboxes, fp, indent=4)