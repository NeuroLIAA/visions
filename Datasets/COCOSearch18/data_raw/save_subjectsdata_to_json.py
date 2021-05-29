import os
import shutil
import json
import numpy as np

""" This script requires that the COCOSearch18 images are in the folder ../images.
    Since the same image can be used for several tasks, those images are renamed as separate files for each task.
    Images are also extracted from the category folder and placed in ../images.
    Only 80% of the dataset is available, so there are some images that are never used in the human trials files.
"""

human_scanpaths_train_file = './coco_search18_fixations_TP_train_split1.json' 
human_scanpaths_valid_file = './coco_search18_fixations_TP_validation_split1.json'

images_dir = '../images/'
save_path  = '../human_scanpaths/'
image_height = 1050
image_width  = 1680

screen_height = 1050
screen_width  = 1650

# TODO: Definir valores, según lo que indique el paper
receptive_height = 54
receptive_width  = 54

max_fixations = 45

if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(human_scanpaths_train_file, 'r') as fp:
    human_scanpaths_train = json.load(fp)

with open(human_scanpaths_valid_file, 'r') as fp:
    human_scanpaths_valid = json.load(fp)

human_scanpaths = human_scanpaths_train + human_scanpaths_valid

images_tasks  = {}
unused_images = 0

initial_fixations_x = []
initial_fixations_y = []

targets_found       = 0
wrong_targets_found = 0
largest_scanpath    = 0
scanpaths_with_shorter_distance_than_receptive_size = 0

subjects = {}
for scanpath in human_scanpaths:
    current_subject = scanpath['subject']

    if current_subject in subjects:
        subject_scanpaths = subjects[current_subject]
    else:
        subject_scanpaths = {}
        subjects[current_subject] = subject_scanpaths

    image_name = scanpath['name']
    task       = scanpath['task']
    
    # Check if the task of the trial is different this time for this image
    if not image_name in images_tasks:
        images_tasks[image_name] = { 'task' : task, 'new_name' : None }
        shutil.move(images_dir + task + '/' + image_name, images_dir + image_name)
    else:
        image_info = images_tasks[image_name]
        while task != image_info['task']:
            # Remove file from subfolder
            image_path = images_dir + task + '/' + image_name
            if os.path.exists(image_path):
                os.remove(image_path)
            # Iterate through dict to define a new name for the file
            new_name = image_info['new_name']
            if new_name is None:
                new_name = str(int(image_name[0]) + 1) + image_name[1:]
                image_info['new_name'] = new_name
                images_tasks[new_name] = { 'task' : task, 'new_name' : None }
                shutil.copyfile(images_dir + image_name, images_dir + new_name)

            image_info = images_tasks[new_name]
            image_name = new_name

    scanpath_x      = scanpath['X']
    scanpath_y      = scanpath['Y']
    scanpath_length = scanpath['length']
    target_bbox     = scanpath['bbox']
    # New target bounding box shape will be [first row, first column, last row, last column]
    target_bbox = [target_bbox[1], target_bbox[0], target_bbox[1] + target_bbox[3], target_bbox[0] + target_bbox[2]]

    if scanpath['correct'] == 1:
        target_found = True
        targets_found += 1
    else:
        target_found = False

    initial_fixations_x.append(scanpath_x[0])
    initial_fixations_y.append(scanpath_y[0])
    
    last_fixation_x = scanpath_x[scanpath_length - 1]
    last_fixation_y = scanpath_y[scanpath_length - 1]
    # Sanity check
    between_bounds = target_bbox[0] <= last_fixation_y + receptive_height and target_bbox[2] >= last_fixation_y - receptive_height and \
                     target_bbox[1] <= last_fixation_x + receptive_width  and target_bbox[3] >= last_fixation_x - receptive_width
    if target_found and not between_bounds:
        print('Subject: ' + str(current_subject) + '; trial: ' + image_name + '. Last fixation doesn\'t fall between target\'s bounds')
        print('Target bbox: ' + str(target_bbox) + '. Last fixation: ' + str((last_fixation_y, last_fixation_x)) + '\n')
        target_found = False
        wrong_targets_found += 1

    if target_found and scanpath_length > largest_scanpath: 
        largest_scanpath = scanpath_length

    if current_subject < 10:
        current_subject_string = '0' + str(current_subject)
    else:
        current_subject_string = str(current_subject)

    # Check for distance between consecutive fixations
    if target_found and scanpath_length > 1:
        fixations = [fix for fix in zip(scanpath_x, scanpath_y)]
        distance_between_consecutive_fixations          = [np.linalg.norm(np.array(fix_1) - np.array(fix_2)) for fix_1, fix_2 in zip(fixations, fixations[1:])]
        shortest_distance_between_consecutive_fixations = min(distance_between_consecutive_fixations)

        fixation_number = distance_between_consecutive_fixations.index(shortest_distance_between_consecutive_fixations)
        shortest_consecutive_fixations_distance = (abs(scanpath_x[fixation_number] - scanpath_x[fixation_number + 1]), abs(scanpath_y[fixation_number] - scanpath_y[fixation_number + 1]))
        if shortest_consecutive_fixations_distance[0] < receptive_height // 2 and shortest_consecutive_fixations_distance[1] < receptive_width // 2:
            scanpaths_with_shorter_distance_than_receptive_size += 1

    subject_scanpaths[image_name] = {'subject' : current_subject_string, 'dataset' : 'COCOSearch18 Dataset', 'image_height' : image_height, 'image_width' : image_width, \
        'screen_height' : screen_height, 'screen_width' : screen_width, 'receptive_height' : receptive_height, 'receptive_width' : receptive_width, 'target_found' : target_found, \
            'target_bbox' : target_bbox, 'X' : scanpath_x, 'Y' : scanpath_y, 'T' : scanpath['T'], 'target_object' : scanpath['task'], 'max_fixations' : max_fixations}

# Save a file for each subject
for subject in subjects:
    if subject < 10:
        subject_string = '0' + str(subject)
    else:
        subject_string = str(subject)

    subject_scanpaths_file = 'subj' + subject_string + '_scanpaths.json'
    with open(save_path + subject_scanpaths_file, 'w') as fp:
        json.dump(subjects[subject], fp, indent=4)

# Clean up unused images
categories = [filename for filename in os.listdir(images_dir) if os.path.isdir(images_dir + filename)]
for category in categories:
    unused_images += len(os.listdir(images_dir + category))
    shutil.rmtree(images_dir + category)


print('Total targets found: ' + str(targets_found) + '. Wrong targets found: ' + str(wrong_targets_found))
print('Number of unused images: ' + str(unused_images))
print('Initial fixation mean: ' + str(round(np.mean(initial_fixations_y), 2), round(np.mean(initial_fixations_x), 2)))
print('Largest target found scanpath: ' + str(largest_scanpath))
print('Scanpaths where saccades have shorter distance than ' + str((receptive_height, receptive_width)) + ': ' + str(scanpaths_with_shorter_distance_than_receptive_size))