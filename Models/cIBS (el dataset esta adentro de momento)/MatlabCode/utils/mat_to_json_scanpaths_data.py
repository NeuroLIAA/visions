from scipy.io import loadmat
import json
import os
import numpy as np


cfg_dir = '../out_models/deepgaze/correlation/a_3_b_4_tam_celda_32/cfg/'
scanpaths_dir= '../out_models/deepgaze/correlation/a_3_b_4_tam_celda_32/scanpath/'
cfg_files = os.listdir(cfg_dir)
scanpaths_files = os.listdir(scanpaths_dir)
save_path = '../../../../Results/cIBS_dataset/cIBS/'

window_size = (32, 32)

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

    target_center_top_left = cfg_info['target_center_top_left'][0][0][0]
    target_center_bot_right = cfg_info['target_center_bot_right'][0][0][0]

    number_of_fixations = len(scanpath)
    last_fixation = scanpath[number_of_fixations - 1]

    between_bounds = (target_center_top_left[0] <= last_fixation[0]) and (target_center_bot_right[0] >= last_fixation[0]) and (target_center_top_left[1] <= last_fixation[1]) and (target_center_bot_right[1] >= last_fixation[1])

    if (number_of_fixations < 16) or between_bounds:
        target_found = True
    else:
        target_found = False

    max_fixations = int(cfg_info['nsaccades_thr'][0][0][0][0]) + 1

    # Convert to pixels
    delta = int(cfg_info['delta'][0][0][0][0])
    scanpath_x = []
    scanpath_y = []
    for i in range(number_of_fixations):
        i_fixation = mapCellToFixation(scanpath[i].astype(int), image_height, image_width, delta)
        scanpath_x.append(int(i_fixation[0]))
        scanpath_y.append(int(i_fixation[1]))

    scanpath_data[image_name] = {"dataset" : "cIBS Dataset", "subject" : "cIBS model", "target_found" : target_found, "X" : scanpath_x, "Y" : scanpath_y, "image_height" : image_height, \
        "image_width" : image_width, "target_object" : "TBD", "max_fixations" : max_fixations}
    
if not(os.path.exists(save_path)):
    os.mkdir(save_path)

with open(save_path + 'Scanpaths.json', 'w') as fp:
    json.dump(scanpath_data, fp, indent = 4)


