from scipy.io import loadmat
import json
import os

subjects_dir = 'sinfo_subj/'
subjects_files = os.listdir(subjects_dir)
save_path = '../human_scanpaths/'

window_size = (32, 32)

targets_found = 0
wrong_targets_found = 0
for subject_file in subjects_files:
    subject_info = loadmat(subjects_dir + subject_file)
    subject_info = subject_info['info_per_subj']

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Processing ' + subject_file)
    print('\n')

    split_subject_filename = subject_file.split('_')
    subject_id = split_subject_filename[len(split_subject_filename) - 1][:-4]
    if (int(subject_id) < 10):
        subject_id = '0' + subject_id

    json_subject = []
    for record in range(len(subject_info[0])):
        image_name   = subject_info['image_name'][0][record][0]
        image_height = int(subject_info['image_size'][0][record][0][0])
        image_width  = int(subject_info['image_size'][0][record][0][1])

        screen_height = int(subject_info['screen_size'][0][record][0][0])
        screen_width  = int(subject_info['screen_size'][0][record][0][1])

        target_bbox = subject_info['target_rect'][0][record][0]
        # Swap values, new order is [lower_row, lower_column, upper_row, upper_column]
        target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3] = target_bbox[1], target_bbox[0], target_bbox[3], target_bbox[2]
        target_found = bool(subject_info['target_found'][0][record][0][0])

        max_fixations = int(subject_info['nsaccades_allowed'][0][record][0][0])
        fix_posX = subject_info['x'][0][record][0]
        fix_posY = subject_info['y'][0][record][0]
        fix_time = subject_info['dur'][0][record][0]

        if (len(fix_posX) == 0):
            print("Subject: " + subject_id + "; stimuli: " + image_name + "; trial: " + str(record + 1) + ". Empty scanpath")
            continue

        number_of_fixations = len(fix_posX)
        last_fixation_X = fix_posX[number_of_fixations - 1]
        last_fixation_Y = fix_posY[number_of_fixations - 1]
        between_bounds = (target_bbox[0] - window_size[0] <= last_fixation_Y) and (target_bbox[2] + window_size[0] >= last_fixation_Y) and (target_bbox[1] - window_size[1] <= last_fixation_X) and (target_bbox[3] + window_size[1] >= last_fixation_X)
        if (target_found):
            if (between_bounds):
                targets_found += 1
            else:
                print("Subject: " + subject_id + "; stimuli: " + image_name + "; trial: " + str(record + 1) + ". Last fixation doesn't match target's bounds")
                print("Target's bounds: " + str(target_bbox) + ". Last fixation: " + str((last_fixation_Y, last_fixation_X)) + '\n')
                wrong_targets_found += 1
                target_found = False

        json_subject.append({ "subject" : subject_id, "image" : image_name, "dataset" : "cIBS Dataset", "image_height" : image_height, "image_width" : image_width, "screen_height" : screen_height, "screen_width" : screen_width, "window_height" : window_size[0], "window_width" : window_size[1], \
            "target_found" : str(target_found), "target_bbox" : target_bbox.tolist(), "X" : fix_posX.tolist(), "Y" : fix_posY.tolist(), "T" : fix_time.tolist(), "split" : "valid", "target_object" : "TBD", "max_fixations" : max_fixations})
    
    if not(os.path.exists(save_path)):
        os.mkdir(save_path)
    subject_json_filename = 'subj' + subject_id + '_scanpaths.json'
    with open(save_path + subject_json_filename, 'w') as fp:
        json.dump(json_subject, fp, indent = 4)

print("Targets found: " + str(targets_found) + ". Wrong targets found: " + str(wrong_targets_found))