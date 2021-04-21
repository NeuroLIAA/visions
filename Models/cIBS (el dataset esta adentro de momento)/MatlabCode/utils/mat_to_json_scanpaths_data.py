from scipy.io import loadmat
import json
import os
import numpy as np
import re
# Helper function to get correct order of stimuli
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

cfg_dir = '../out_models/deepgaze/correlation/a_3_b_4_tam_celda_160/cfg/'
scanpaths_dir= '../out_models/deepgaze/correlation/a_3_b_4_tam_celda_160/scanpath/'
cfg_files = sorted_alphanumeric(os.listdir(cfg_dir))
save_path = '../../../../Results/IVSN_dataset/cIBS/'

def mapCellToFixation(cell, img_height, img_width, delta):
    img_size = np.array([img_height, img_width])
    grid_size = img_size // delta
    offset = (img_size - grid_size * delta) / 2

    fixation = cell * delta + offset
    return fixation

scanpath_data = dict()
for cfg_file in cfg_files:
    if cfg_file == 'time.mat':
        continue

    cfg_info = loadmat(cfg_dir + cfg_file)
    cfg_split_file_name = cfg_file.split('_')
    cfg_id = cfg_split_file_name[len(cfg_split_file_name) - 1][:-4]
    
    scanpath = loadmat(scanpaths_dir + 'scanpath_' + cfg_id + '.mat')
    cfg_info = cfg_info['cfg']
    scanpath = scanpath['scanpath']
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Processing ' + cfg_file)

    image_name   = cfg_info['imgname'][0][0][0]
    image_height = int(cfg_info['image_size'][0][0][0][0])
    image_width  = int(cfg_info['image_size'][0][0][0][1])

    delta = int(cfg_info['delta'][0][0][0][0])

    target_center_top_left = cfg_info['target_center_top_left'][0][0][0]
    target_center_bot_right = cfg_info['target_center_bot_right'][0][0][0]
    target_size = cfg_info['target_size'][0][0][0]

    target_center_top_left_pixels  = mapCellToFixation(target_center_top_left.astype(int), image_height, image_width, delta)
    target_center_bot_right_pixels = mapCellToFixation(target_center_bot_right.astype(int), image_height, image_width, delta)
    target_bbox = [int(target_center_top_left_pixels[0]), int(target_center_top_left_pixels[1]), int(target_center_bot_right_pixels[0]), int(target_center_bot_right_pixels[1])]

    number_of_fixations = len(scanpath)
    last_fixation = scanpath[number_of_fixations - 1]

    between_bounds = (target_center_top_left[0] <= last_fixation[0]) and (target_center_bot_right[0] >= last_fixation[0]) and (target_center_top_left[1] <= last_fixation[1]) and (target_center_bot_right[1] >= last_fixation[1])

    if (number_of_fixations < 31) or between_bounds:
        target_found = True
    else:
        target_found = False

    max_fixations = int(cfg_info['nsaccades_thr'][0][0][0][0]) + 1

    # Convert to pixels
    scanpath_x = []
    scanpath_y = []
    for i in range(number_of_fixations):
        i_fixation = mapCellToFixation(scanpath[i].astype(int), image_height, image_width, delta)
        scanpath_x.append(int(i_fixation[0]) - 1)
        scanpath_y.append(int(i_fixation[1]) - 1)

    scanpath_data[image_name] = {"subject" : "cIBS model", "dataset" : "IVSN Natural Design Dataset", "image_height" : image_height, "image_width" : image_width, "receptive_height" : delta, "receptive_width" : delta, \
        "target_found" : target_found, "target_bbox": target_bbox, "X" : scanpath_x, "Y" : scanpath_y, "target_object" : "TBD", "max_fixations" : max_fixations}
    
if not(os.path.exists(save_path)):
    os.mkdir(save_path)

with open(save_path + 'Scanpaths_delta_160.json', 'w') as fp:
    json.dump(scanpath_data, fp, indent = 4)


