import numpy as np

base_data_path = "dataset/"
def get_data_paths(task, i):
    stim_path = base_data_path + task + '/stimuli/img' + str(i+1).zfill(3) + '.jpg'
    gt_path = base_data_path + task + '/gt/gt' + str(i+1) + '.jpg'
    tar_path = base_data_path + task + '/target/t' + str(i+1).zfill(3) + '.jpg'

    return stim_path, gt_path, tar_path

def get_exp_info(task):
    s_ratio = None
    if task == "NaturalDesign":
        eye_res = 736
        stim_shape = (736, 896, 3)
        s_ratio = (1024/stim_shape[0], 1280/stim_shape[1])
        tar_shape = (32, 32, 3)
        ior_size = 2*tar_shape[0]
        NumStimuli = 240
        NumFix = 66
        weight_pattern = 'l'
        gt_mask = None
        fix = None
    elif task == "Waldo":
        eye_res = 736
        stim_shape = (736, 896, 3)
        s_ratio = (1024/stim_shape[0], 1280/stim_shape[1])
        tar_shape = (32, 32, 3)
        ior_size = 2*tar_shape[0]
        NumStimuli = 67
        NumFix = 81
        gt_mask = None
        fix = None
        weight_pattern = 'l'
    elif task == "ObjArr":
        eye_res = 544
        stim_shape = (544, 544, 3)
        tar_shape = (64, 64, 3)
        s_ratio = None
        ior_size = 45
        NumStimuli = 300
        NumFix = 7
        weight_pattern = 'l'
        gt_mask = np.load("dataset/" + task + "/gt_mask.npy")
        fix = [[640, 512],
               [365, 988],
               [90, 512],
               [365, 36],
               [915, 36],
               [1190, 512],
               [915, 988]]
        
    exp_info = {}
    exp_info['eye_res'] = eye_res
    exp_info['stim_shape'] = stim_shape
    exp_info['tar_shape'] = tar_shape
    exp_info['ior_size'] = ior_size
    exp_info['NumStimuli'] = NumStimuli
    exp_info['NumFix'] = NumFix
    exp_info['gt_mask'] = gt_mask
    exp_info['fix'] = fix
    exp_info['bg_value'] = 0
    exp_info['rev_img_flag'] = 0
    exp_info['deg2px'] = 30
    exp_info['weight_pattern'] = weight_pattern
    exp_info['dog_size'] = [[3, 0], [5, 1]]
    exp_info['s_ratio'] = s_ratio

    return exp_info
