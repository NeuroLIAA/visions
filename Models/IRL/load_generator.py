import torch
import numpy as np
import json
from os.path import join
from irl_dcb.utils  import load
from irl_dcb.models import LHF_Policy_Cond_Small
from irl_dcb.config import JsonConfig

device = torch.device('cpu')
hparams = JsonConfig('hparams/coco_search18.json')
input_size = 134 # number of belief maps
 
with open(join('dataset_root/', 'human_scanpaths_TP_trainval_train.json')) as json_file:
        human_scanpaths_train = json.load(json_file)
human_scanpaths_train = list(filter(lambda x: x['correct'] == 1, human_scanpaths_train))

cat_names = list(np.unique([x['task'] for x in human_scanpaths_train]))
catIds = dict(zip(cat_names, list(range(len(cat_names)))))

task_eye = torch.eye(len(catIds)).to(device)

generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(catIds), task_eye,
                                      input_size).to(device)

load('best', generator, 'generator', pkg_dir='trained_models/', device=device)

print(generator)