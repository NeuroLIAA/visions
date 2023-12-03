import shutil
import utils
import numpy as np
from pathlib import Path
from scipy.io import loadmat

dataset_name = 'People Dataset'
eye_tracking_dir = Path('eyeData')
eye_tracking_files = list(eye_tracking_dir.glob('*.mat'))

targets_dir     = Path('../targets/')
scanpaths_dir   = Path('../human_scanpaths/')
trials_properties_file = Path('../trials_properties.json')
target_img = 'pedestrian.jpg'

image_height, image_width  = 600, 800
screen_height, screen_width = 768, 1024
# Estimated from the paper with a fovea size of 2 degrees
receptive_size = (42, 42)
max_fixations = 16
# Mean of subjects' initial fixations (image center)
initial_fixation = (302, 398)

number_of_trials    = 0
targets_found       = 0
wrong_targets_found = 0
cropped_scanpaths   = 0
collapsed_scanpaths = 0
collapsed_fixations = 0
trivial_scanpaths   = 0

trials_properties = []
subjects_trials = {}
subjects_mapping = {}
for img_data_file in eye_tracking_files:
    img_data = loadmat(img_data_file, simplify_cells=True)[img_data_file.stem]
    if img_data['target'] != 'TP':
        continue
    img_name = img_data['imgName']
    target_bbox = img_data['targetInfo']['targetBox']
    bbox_upper = target_bbox[:1].flatten()
    bbox_lower = target_bbox[1:].flatten()
    # Upper row, upper column, lower row, lower column
    target_bbox = [int(bbox_upper[1]), int(bbox_upper[0]), int(bbox_lower[1]), int(bbox_lower[0])]
    
    for i, subj in enumerate(img_data['subdata'], 1):
        name = subj['subName']
        nfixs = subj['Nfixs']
        scanpath = subj['fixXY']
        scanpath_t = subj['fixDur']
        gotbox = bool(subj['gotBox'])
        if nfixs == 1:
            scanpath = [scanpath]
            scanpath_t = [subj['fixDur']]
        scanpath_x, scanpath_y = [fix[0] for fix in scanpath], [fix[1] for fix in scanpath]
        
        
        # Collapse consecutive fixations which are closer than receptive_size / 2
        scanpath_x, scanpath_y = utils.collapse_fixations(scanpath_x, scanpath_y, receptive_size)
        if len(scanpath_x) < nfixs:
            collapsed_scanpaths += 1
            collapsed_fixations += nfixs - len(scanpath_x)

        scanpath_x, scanpath_y = [round(x, 1) for x in scanpath_x], [round(y, 1) for y in scanpath_y]
        scanpath_t = [int(round(t, 1)) for t in scanpath_t]
        cropped_scanpath_len = len(scanpath_x)
        # Crop scanpaths as soon as a fixation falls between the target's bounding box
        target_found, scanpath_x, scanpath_y = utils.crop_scanpath(scanpath_x, scanpath_y, target_bbox, receptive_size, (image_height, image_width))
        if len(scanpath_x) == 1:
            trivial_scanpaths += 1
            continue
        if target_found: 
            targets_found += 1
        if len(scanpath_x) < cropped_scanpath_len: 
            cropped_scanpaths += 1
            
        if not target_found and gotbox:
            wrong_targets_found += 1
        
        if name not in subjects_mapping:
            subjects_mapping[name] = i
            subj_id = i
        else:
            subj_id = subjects_mapping[name]
        subj_id_str = f'{subj_id:02d}'
        if not name in subjects_trials:
            subjects_trials[name] = {}
            
        number_of_trials += 1
        
        subjects_trials[name][img_name] = {'subject': subj_id_str, 'dataset': dataset_name, 'image_height': image_height, 'image_width': image_width, \
        'screen_height': screen_height, 'screen_width': screen_width, 'receptive_height': receptive_size[0], 'receptive_width': receptive_size[1], 'target_found' : target_found, \
            'target_bbox': target_bbox, 'X' : scanpath_x, 'Y': scanpath_y, 'T': list(scanpath_t), 'target_object': 'person', 'max_fixations': max_fixations}

    target_name = img_name[:-4] + '_target' + img_name[-4:]
    target_matched_row, target_matched_column = target_bbox[0], target_bbox[1]
    target_height, target_width = target_bbox[2] - target_matched_row, target_bbox[3] - target_matched_column
    trials_properties.append({'image': img_name, 'target': target_name, 'dataset': dataset_name, \
        'target_matched_row': target_matched_row, 'target_matched_column': target_matched_column, 'target_height': target_height, 'target_width': target_width, \
            'image_height': image_height, 'image_width': image_width, 'initial_fixation_row': initial_fixation[0], 'initial_fixation_column': initial_fixation[1], \
                'target_object': 'person'})
    
    shutil.copyfile(target_img, targets_dir / target_name)

utils.save_to_json(trials_properties_file, trials_properties)

for subject in subjects_trials:
    subject_trials = subjects_trials[subject]
    subj_id = subjects_mapping[subject]
    subj_file = f'subj{subj_id:02d}_scanpaths.json'
    utils.save_to_json(scanpaths_dir / subj_file, subject_trials)

print('Total targets found: ' + str(targets_found) + '/' + str(number_of_trials) + '. Wrong targets found: ' + str(wrong_targets_found))
print("Collapsed scanpaths (discretized in size " + str(receptive_size) + ") : " + str(collapsed_scanpaths))
print("Number of fixations collapsed: " + str(collapsed_fixations))
print('Cropped scanpaths (target found earlier): ' + str(cropped_scanpaths))
print('Trivial scanpaths (length one): ' + str(trivial_scanpaths))