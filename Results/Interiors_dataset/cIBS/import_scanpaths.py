import json
from os import path, listdir
from scipy.io import loadmat

def between_bounds(target_bbox, fix_y, fix_x, receptive_size):
    return target_bbox[0] <= fix_y + receptive_size[0] // 2 and target_bbox[2] >= fix_y - receptive_size[0] // 2 and \
            target_bbox[1] <= fix_x + receptive_size[1] // 2 and target_bbox[3] >= fix_x - receptive_size[1] // 2

scanpaths = listdir('cibs/scanpath')
scanpaths_JSON = {}
for scanpath_file in scanpaths:
    cfg_file = 'cfg' + scanpath_file[8:]
    scanpath_mat = loadmat(path.join('cibs/scanpath', scanpath_file))
    scanpath_cfg = loadmat(path.join('cibs/cfg', cfg_file))
    
    cfg = scanpath_cfg['cfg']
    scanpath_mat = scanpath_mat['scanpath']
    scanpath_X   = [int(scanpath_mat[i][1] - 1) for i in range(len(scanpath_mat))]
    scanpath_Y   = [int(scanpath_mat[i][0] - 1) for i in range(len(scanpath_mat))]
    target_bbox  = [int(cfg['target_center_top_left'][0][0][0][0] - 1), int(cfg['target_center_top_left'][0][0][0][1] - 1), \
        int(cfg['target_center_bot_right'][0][0][0][0] - 1), int(cfg['target_center_bot_right'][0][0][0][1] - 1)]
    receptive_size = [1, 1]
    target_found = between_bounds(target_bbox, scanpath_Y[len(scanpath_Y) - 1], scanpath_X[len(scanpath_X) - 1], receptive_size)

    max_fixations = int(cfg['nsaccades_thr'][0][0][0][0] + 1)

    image_name = cfg['imgname'][0][0][0]

    scanpaths_JSON[image_name] = {'subject' : 'cIBS Model', 'dataset' : 'Interiors Dataset', 'image_height' : 24, 'image_width' : 32, \
        'receptive_height' : 1, 'receptive_width' : 1, 'target_found' : bool(target_found), 'target_bbox' : target_bbox, \
                 'X' : scanpath_X, 'Y' : scanpath_Y, 'target_object' : 'TBD', 'max_fixations' : max_fixations
        }

with open('Scanpaths_MATLAB.json', 'w') as fp:
    json.dump(scanpaths_JSON, fp, indent=4)