import os
import json

human_scanpaths_train_file = './coco_search18_fixations_TP_train_split1.json' 
human_scanpaths_valid_file = './coco_search18_fixations_TP_validation_split1.json'

save_path = '../human_scanpaths/'
image_height = 1050
image_width  = 1680

screen_height = 1050
screen_width  = 1650

# TODO: Definir valores, según lo que indique el paper
# receptive_height = ??
# receptive_width  = ??

# max_fixations = ??

if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(human_scanpaths_train_file, 'r') as fp:
    human_scanpaths_train = json.load(fp)

with open(human_scanpaths_valid_file, 'r') as fp:
    human_scanpaths_valid = json.load(fp)

human_scanpaths = human_scanpaths_train + human_scanpaths_valid

subjects = {}
targets_found       = 0
wrong_targets_found = 0
for scanpath in human_scanpaths:
    current_subject = scanpath['subject']

    if current_subject in subjects:
        subject_scanpaths = subjects[current_subject]
    else:
        subject_scanpaths = {}
        subjects[current_subject] = subject_scanpaths

    image_name      = scanpath['name']
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
    
    last_fixation_x = scanpath_x[scanpath_length - 1]
    last_fixation_y = scanpath_y[scanpath_length - 1]
    # Sanity check
    between_bounds = target_bbox[0] <= last_fixation_y and target_bbox[2] >= last_fixation_y and \
                     target_bbox[1] <= last_fixation_x and target_bbox[3] >= last_fixation_x
    if target_found and not between_bounds:
        print('Subject: ' + str(current_subject) + '; trial: ' + image_name + '. Last fixation doesn\'t fall between target\'s bounds')
        print('Target bbox: ' + str(target_bbox) + '. Last fixation: ' + str((last_fixation_y, last_fixation_x)) + '\n')
        target_found   = False
        wrong_targets_found += 1

    if current_subject < 10:
        current_subject_string = '0' + str(current_subject)
    else:
        current_subject_string = str(current_subject)

    subject_scanpaths[image_name] = {'subject' : current_subject_string, 'dataset' : 'COCOSearch18 Dataset', 'image_height' : image_height, 'image_width' : image_width, \
        'screen_height' : screen_height, 'screen_width' : screen_width, 'receptive_height' : 'definir!!', 'receptive_width' : 'definir!!', 'target_found' : target_found, \
            'target_bbox' : target_bbox, 'X' : scanpath_x, 'Y' : scanpath_y, 'T' : scanpath['T'], 'target_object' : scanpath['task'], 'max_fixations' : 'definir!!'}

# Save a file for each subject
for subject in subjects:
    if subject < 10:
        subject_string = '0' + str(subject)
    else:
        subject_string = str(subject)

    subject_scanpaths_file = 'subj' + subject_string + '_scanpaths.json'
    with open(save_path + subject_scanpaths_file, 'w') as fp:
        json.dump(subjects[subject], fp, indent=4)


print('Total targets found: ' + str(targets_found) + '. Wrong targets found: ' + str(wrong_targets_found))