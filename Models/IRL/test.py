
"""Test script.
Usage:
  test.py <hparams> <checkpoint_dir> <dataset_root> [--cuda=<id>]
  test.py -h | --help
Options:
  -h --help     Show this screen.
  --cuda=<id>   id of the cuda device [default: 0].
"""

import os
import torch
import numpy as np
import json
from tqdm import tqdm
from docopt import docopt
from os.path import join
from dataset import process_data
from irl_dcb.config import JsonConfig
from torch.utils.data import DataLoader
from irl_dcb.models import LHF_Policy_Cond_Small
from irl_dcb.environment import IRL_Env4LHF
from irl_dcb import metrics
from irl_dcb import utils
torch.manual_seed(42619)
np.random.seed(42619)


def gen_scanpaths(generator,
                  env_test,
                  test_img_loader,
                  patch_num,
                  max_traj_len,
                  im_w,
                  im_h,
                  num_sample=10):
    all_actions = []
    for i_sample in range(num_sample):
        progress = tqdm(test_img_loader,
                        desc='trial ({}/{})'.format(i_sample + 1, num_sample))
        for i_batch, batch in enumerate(progress):
            env_test.set_data(batch)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            with torch.no_grad():
                env_test.reset()
                trajs = utils.collect_trajs(env_test,
                                            generator,
                                            patch_num,
                                            max_traj_len,
                                            is_eval=True,
                                            sample_action=True)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(env_test.batch_size)])
            
    scanpaths = utils.actions2scanpaths(all_actions, patch_num, im_w, im_h)
    utils.cutFixOnTarget(scanpaths, bbox_annos)
        

    return scanpaths


if __name__ == '__main__':
    args = docopt(__doc__)
    device = torch.device('cpu')
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    #https://drive.google.com/drive/folders/1spD2_Eya5S5zOBO3NKILlAjMEC3_gKWc de ahí se baja todo lo que iría en dataset_root
    checkpoint = args["<checkpoint_dir>"]
    hparams = JsonConfig(hparams)

    # dir of pre-computed beliefs
    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')
    data_name = '{}x{}'.format(hparams.Data.im_w, hparams.Data.im_h)

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()
    #hay que ver como computar esto, ni idea

    # load ground-truth human scanpaths
    with open(join(dataset_root,
                   'human_scanpaths_TP_trainval_train.json')) as json_file:
                   #esto dependerá del dataset, pero solo sirve para calcular métricas
        human_scanpaths = json.load(json_file)

    human_scanpaths = list(filter(lambda x: x['correct'] == 1, human_scanpaths))



    # loading test fixation clusters (for computing sequence score)
    #fix_clusters = np.load(join('./data', 'clusters.npy'), allow_pickle=True).item()
    #esto lo saque porque no se a que se refiere y lo pedía 

    # process fixation data
    dataset = process_data(human_scanpaths,
                           DCB_dir_HR,
                           DCB_dir_LR,
                           bbox_annos,
                           hparams,
                           is_testing=True)
    img_loader = DataLoader(dataset['img_test'],
                            batch_size=64,
                            shuffle=False,
                            num_workers=4)
    print('num of test images =', len(dataset['img_test']))

    # load trained model
    input_size = 134  # number of belief maps
    task_eye = torch.eye(len(dataset['catIds'])).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(dataset['catIds']), task_eye,
                                      input_size).to(device)

    
    utils.load('best', generator, 'generator', pkg_dir=checkpoint, device = device)
    
    generator.eval()

    # build environment
    env_test = IRL_Env4LHF(hparams.Data,
                           max_step=hparams.Data.max_traj_length,
                           mask_size=hparams.Data.IOR_size,
                           status_update_mtd=hparams.Train.stop_criteria,
                           device=device,
                           inhibit_return=True)

    # generate scanpaths
    print('sample scanpaths (10 for each testing image)...')
    predictions = gen_scanpaths(generator,
                                env_test,
                                img_loader,
                                hparams.Data.patch_num,
                                hparams.Data.max_traj_length,
                                hparams.Data.im_w,
                                hparams.Data.im_h,
                                num_sample=1)

    print('evaluating model...')
    # evaluate predictions
    mean_cdf, _ = utils.compute_search_cdf(predictions, bbox_annos,
                                           hparams.Data.max_traj_length)

    # scanpath ratio
    sp_ratio = metrics.compute_avgSPRatio(predictions, bbox_annos,
                                          hparams.Data.max_traj_length)

    # probability mismatch
    prob_mismatch = metrics.compute_prob_mismatch(mean_cdf,
                                                  dataset['human_mean_cdf'])

    # TFP-AUC
    tfp_auc = metrics.compute_cdf_auc(mean_cdf)

    # sequence score
    seq_score = 0#metrics.get_seq_score(predictions, fix_clusters,
                                      #hparams.Data.max_traj_length)
    #lo saqué porque pedía los clusters

    # multimatch
    mm_score = metrics.compute_mm(dataset['gt_scanpaths'], predictions,
                                  hparams.Data.im_w, hparams.Data.im_h)

    

    #para que no putee el json al dumpear scanpaths xd
    for prediction in predictions:
        prediction['X']= list(prediction['X'])
        prediction['Y']= list(prediction['Y'])
    predictions = list(predictions)

    
    # print and save outputs
    print('results:')
    results = {
        'scanpaths': predictions,
        'cdf': list(mean_cdf),
        'sp_ratios': sp_ratio,
        'probability_mismatch': prob_mismatch,
        'TFP-AUC': tfp_auc,
        'sequence_score': seq_score,
        'multimatch': list(mm_score)
    }

    results = JsonConfig(results)
    save_path = join(checkpoint, '../results/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    results.dump(save_path)
    print('results successfully saved to {}'.format(save_path))