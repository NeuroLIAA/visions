#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0, "./")

import numpy as np
from vs_model import VisualSearchModel as VisualSearchModel
from tqdm import tqdm
from data_utils import get_data_paths, get_exp_info
import tensorflow as tf
import time
from pathlib import Path

t_start = time.time()

physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)

vgg_model_path = "pretrained_model/vgg16_imagenet_filters.h5"
base_data_path = "dataset/"

eccParam = {}
eccParam['rf_min'] = [2]*5
eccParam['stride'] = [2]*5
eccParam['ecc_slope'] = [0, 0, 3.5*0.02, 8*0.02, 16*0.02]
eccParam['deg2px'] = [round(30.0), round(30.0/2), round(30.0/4), round(30.0/8), round(30.0/16)]
eccParam['fovea_size'] = 4
eccParam['rf_quant'] = 1
eccParam['pool_type'] = 'avg'

ecc_models = []

# eccNET
for out_layer in [[1, 1, 1]]:
    model_desc = {'eccParam': eccParam,
                  'ecc_depth': 5,
                  'out_layer': out_layer,
                  'comp_layer': 'diff',
                  'vgg_model_path': vgg_model_path,
                  'model_subname': ""}

    ecc_models.append(model_desc)

for model_desc in ecc_models:
    vsm = VisualSearchModel(model_desc)
    print(vsm.model_name)

    for task in ["ObjArr", "Waldo", "NaturalDesign"][2:]:
        exp_info = get_exp_info(task)
        vsm.load_exp_info(exp_info, corner_bias=16*4*1)

        NumStimuli = exp_info['NumStimuli']
        NumFix = exp_info['NumFix']

        data = np.zeros((NumStimuli, NumFix, 2))
        I_data = np.zeros((NumStimuli, 1), dtype=int)
        CP = np.zeros((NumStimuli, NumFix), dtype=int)

        for i in tqdm(range(NumStimuli), desc=task):
            stim_path, gt_path, tar_path = get_data_paths(task, i)
            saccade = vsm.start_search(stim_path, tar_path, gt_path)

            j = saccade.shape[0]
            I_data[i, 0] = min(NumFix, j)
            if j < NumFix+1:
                CP[i, j-1] = 1
            data[i, :min(NumFix, j), 0] = saccade[:, 0].reshape((-1,))[:min(NumFix, j)]
            data[i, :min(NumFix, j), 1] = saccade[:, 1].reshape((-1,))[:min(NumFix, j)]

        if exp_info['s_ratio'] is not None:
            data[:, :, 0] = data[:, :, 0]*exp_info['s_ratio'][0]
            data[:, :, 1] = data[:, :, 1]*exp_info['s_ratio'][1]
        
        save_path = Path('out', task)
        if not save_path.exists(): save_path.mkdir(parents=True)
        cp_path = save_path / 'cp'
        cp_path.mkdir(exist_ok=True)
        ifix_path = save_path / 'ifix'
        ifix_path.mkdir(exist_ok=True)
        fix_path = save_path / 'fix'
        fix_path.mkdir(exist_ok=True)
        np.save(str(cp_path / ('CP_' + vsm.model_name + '.npy')), CP)
        np.save(str(ifix_path / ('I_' + vsm.model_name + '.npy')), I_data)
        np.save(str(fix_path / (vsm.model_name + '.npy')), data)
        breakpoint()

print("Total time taken:", time.time()-t_start)


# In[ ]:
