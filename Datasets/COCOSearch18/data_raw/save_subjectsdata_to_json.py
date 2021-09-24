import os
import shutil
import json
import numpy as np
from skimage import io, transform

""" This script requires that the COCOSearch18 images are in the folder ../images.
    Since the same image can be used for several tasks, those images are renamed as separate files for each task.
    Images are also extracted from the category folder and placed in ../images.
    Only 80% of the dataset is available, so there are some images that are never used in the human trials files.
"""

def rescale_coordinate(value, old_size, new_size):
    return (value / old_size) * new_size

def between_bounds(target_bbox, fix_y, fix_x, receptive_size):
    return target_bbox[0] <= fix_y + receptive_size[0] and target_bbox[2] >= fix_y - receptive_size[0] and \
                     target_bbox[1] <= fix_x + receptive_size[1]  and target_bbox[3] >= fix_x - receptive_size[1]

human_scanpaths_train_file = './coco_search18_fixations_TP_train_split1.json' 
human_scanpaths_valid_file = './coco_search18_fixations_TP_validation_split1.json'

images_dir      = '../images/'
targets_dir     = '../templates/'
scanpaths_dir   = '../human_scanpaths/'
image_height = 1050
image_width  = 1680

screen_height = 1050
screen_width  = 1650

# Estimated value from IRL's model patch size
receptive_height = 52
receptive_width  = 52

max_fixations = 45  

if not os.path.exists(scanpaths_dir):
    os.mkdir(scanpaths_dir)

with open(human_scanpaths_train_file, 'r') as fp:
    human_scanpaths_train = json.load(fp)

with open(human_scanpaths_valid_file, 'r') as fp:
    human_scanpaths_valid = json.load(fp)

human_scanpaths = human_scanpaths_train + human_scanpaths_valid

images_tasks  = {}
unused_images = 0

initial_fixations_x = []
initial_fixations_y = []
# Computed from previous iterations
initial_fixation = (509, 816)

targets_found       = 0
wrong_targets_found = 0
largest_scanpath    = 0
cropped_scanpaths   = 0
scanpaths_with_shorter_distance_than_receptive_size = 0

trials_properties = []
trials_processed  = []

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

    # Crop scanpaths as soon as a fixation falls between the target's bounding box
    for index, fixation in enumerate(zip(scanpath_y, scanpath_x)):
        if between_bounds(target_bbox, fixation[0], fixation[1], (receptive_height, receptive_width)):
            scanpath_x = scanpath_x[:index + 1]
            scanpath_y = scanpath_y[:index + 1]
            cropped_scanpaths += 1
            if not target_found:
                target_found   = True
                targets_found += 1
            break
    
    scanpath_length = len(scanpath_x)
    last_fixation_x = scanpath_x[scanpath_length - 1]
    last_fixation_y = scanpath_y[scanpath_length - 1]
    # Sanity check
    if target_found and not between_bounds(target_bbox, last_fixation_y, last_fixation_x, (receptive_height, receptive_width)):
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

    if not image_name in trials_processed:
        # Save trial info
        target_name = image_name[:-4] + '_template' + image_name[-4:]
        target_matched_row    = target_bbox[0]
        target_matched_column = target_bbox[1]
        target_height         = target_bbox[2] - target_matched_row
        target_width          = target_bbox[3] - target_matched_column
        trials_properties.append({'image' : image_name, 'target' : target_name, 'dataset' : 'COCOSearch18 Dataset', \
            'target_matched_row' : target_matched_row, 'target_matched_column' : target_matched_column, 'target_height' : target_height, 'target_width' : target_width, \
                'image_height' : image_height, 'image_width' : image_width, 'initial_fixation_row' : initial_fixation[0], 'initial_fixation_column' : initial_fixation[1], \
                    'target_object' : task})

        # Crop target
        image    = io.imread(images_dir + image_name)
        template = image[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]]
        if not os.path.exists(targets_dir):
            os.mkdir(targets_dir)

        io.imsave(targets_dir + target_name, template, check_contrast=False)

        trials_processed.append(image_name)

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

# Save trials properties
with open('../trials_properties.json', 'w') as fp:
    json.dump(trials_properties, fp, indent=4)

# Save a human_scanpaths file for each subject
for subject in subjects:
    if subject < 10:
        subject_string = '0' + str(subject)
    else:
        subject_string = str(subject)

    subject_scanpaths_file   = 'subj' + subject_string + '_scanpaths.json'
    with open(scanpaths_dir + subject_scanpaths_file, 'w') as fp:
        json.dump(subjects[subject], fp, indent=4)

# Clean up unused images
categories = [filename for filename in os.listdir(images_dir) if os.path.isdir(images_dir + filename)]
for category in categories:
    unused_images += len(os.listdir(images_dir + category))
    shutil.rmtree(images_dir + category)

initial_fixation_mean = (round(np.mean(initial_fixations_y)), round(np.mean(initial_fixations_x)))

print('Total targets found: ' + str(targets_found) + '. Wrong targets found: ' + str(wrong_targets_found))
print('Initial fixation mean: ' + str(initial_fixation_mean))
print('Number of unused images: ' + str(unused_images))
print('Largest target found scanpath: ' + str(largest_scanpath))
print('Number of cropped scanpaths: ' + str(cropped_scanpaths))
print('Scanpaths where saccades have shorter distance than ' + str((receptive_height, receptive_width)) + ': ' + str(scanpaths_with_shorter_distance_than_receptive_size))