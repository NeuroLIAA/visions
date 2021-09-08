
""" Script which runs the IRL model on a given dataset
Usage:
  main.py -dataset <dataset_name>
"""

import torch
import numpy as np
import json
import argparse
import constants
from tqdm import tqdm
from os import path, cpu_count
from dataset import process_eval_data, process_trials, load_human_scanpaths
from irl_dcb.config import JsonConfig
from torch.utils.data import DataLoader
from irl_dcb.models import LHF_Policy_Cond_Small
from irl_dcb.environment import IRL_Env4LHF
from irl_dcb import utils

torch.manual_seed(42619)
np.random.seed(42619)

def run_visualsearch(dataset_name, human_subject, trained_models_dir, hparams, device):
    dcbs_path = path.join(constants.DCBS_PATH, dataset_name)
    # Dir of high and low res belief maps
    DCB_dir_HR = path.join(dcbs_path, 'DCBs/HR/')
    DCB_dir_LR = path.join(dcbs_path, 'DCBs/LR/')
    
    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)

    dataset_info      = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))
    trials_properties = utils.load_dict_from_json(path.join(dataset_path, 'trials_properties.json'))
    
    images_dir     = path.join(dataset_path, dataset_info['images_dir'])
    new_image_size = (hparams.Data.im_h, hparams.Data.im_w)
    grid_size      = (hparams.Data.patch_num[1], hparams.Data.patch_num[0])
    patch_size     = (hparams.Data.patch_size[1], hparams.Data.patch_size[0])

    # For computing different metrics
    human_scanpaths_dir = path.join(dataset_path, dataset_info['scanpaths_dir'])
    human_scanpaths     = load_human_scanpaths(human_scanpaths_dir, human_subject, grid_size)
    max_len_human_scanpath = utils.get_max_scanpath_length(utils.scanpaths_to_list(human_scanpaths))

    hparams.Data.max_traj_length = dataset_info['max_scanpath_length'] - 1

    # Process trials, creating belief maps when necessary, and get target's bounding box for each trial
    bbox_annos = process_trials(trials_properties, images_dir, human_scanpaths, new_image_size, grid_size, DCB_dir_HR, DCB_dir_LR)

    # Get categories and load image data
    dataset = process_eval_data(trials_properties, human_scanpaths, DCB_dir_HR, DCB_dir_LR, bbox_annos, grid_size, hparams)
    
    batch_size = min(len(trials_properties), 64)
    img_loader = DataLoader(dataset['img_test'],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=cpu_count())

    print('Number of images: ', len(dataset['img_test']))

    # Load trained model
    task_eye  = torch.eye(len(dataset['catIds'])).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(dataset['catIds']), task_eye,
                                      constants.NUMBER_OF_BELIEF_MAPS).to(device)
    
    utils.load('best', generator, 'generator', pkg_dir=trained_models_dir, device=device)
    generator.eval()

    # If following a subject's scanpaths, those vary in length, so we set an upper limit
    if human_scanpaths:
        max_step = max_len_human_scanpath
    else:
        max_step = hparams.Data.max_traj_length
    # Build environment
    env_test = IRL_Env4LHF(hparams.Data,
                           max_step=max_step,
                           mask_size=hparams.Data.IOR_size,
                           status_update_mtd=hparams.Train.stop_criteria,
                           device=device,
                           inhibit_return=True,
                           init_mtd='manual')

    # Generate scanpaths
    print('Generating scanpaths...')
    predictions = gen_scanpaths(generator,
                                env_test,
                                img_loader,
                                bbox_annos,
                                hparams.Data.patch_num,
                                hparams.Data.max_traj_length,
                                hparams.Data.im_w,
                                hparams.Data.im_h,
                                human_scanpaths)

    #### TODO: Guardar scanpaths humanos reescalados
    output_path = path.join(constants.RESULTS_PATH, dataset_name + '_dataset' + '/IRL/')
    utils.save_scanpaths(output_path, predictions)

def gen_scanpaths(generator, env_test, test_img_loader, bbox_annos, patch_num, max_traj_len, im_w, im_h, human_scanpaths, num_sample=1):
    all_actions = []
    for i_sample in range(num_sample):
        progress = tqdm(test_img_loader)
        for i_batch, batch in enumerate(progress):
            env_test.set_data(batch)
            img_names_batch   = batch['img_name']
            cat_names_batch   = batch['cat_name']
            initial_fix_batch = batch['init_fix']
            human_scanpaths_batch = []
            max_traj_len_batch    = max_traj_len
            if human_scanpaths:
                human_scanpaths_batch = [human_scanpaths[image_name] for image_name in img_names_batch]
                max_traj_len_batch    = utils.get_max_scanpath_length(human_scanpaths_batch) - 1
            with torch.no_grad():
                env_test.reset()
                trajs = utils.collect_trajs(env_test,
                                            generator,
                                            img_names_batch,
                                            patch_num,
                                            max_traj_len_batch,
                                            human_scanpaths_batch,
                                            is_eval=True,
                                            sample_action=True)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i], initial_fix_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(env_test.batch_size)])
            
    scanpaths = utils.actions2scanpaths(all_actions, patch_num, im_w, im_h, dataset_name, hparams.Data.patch_size[0], max_traj_len)
    utils.cutFixOnTarget(scanpaths, bbox_annos)

    return scanpaths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the IRL visual search model')
    parser.add_argument('-dataset', type=str, help='Name of the dataset on which to run the model. Value must be one of cIBS, COCOSearch18, IVSN or MCS.')
    parser.add_argument('--h', '--human_subject', type=int, default=None, help='Human subject on which the model will follow its scanpaths, saving the probability map for each saccade.\
         Useful for computing different metrics. See "Kümmerer, M. & Bethge, M. (2021), State-of-the-Art in Human Scanpath Prediction" for more information')
    args = parser.parse_args()

    device             = torch.device('cpu')
    dataset_name       = args.dataset
    human_subject      = args.h
    trained_models_dir = 'trained_models/'
    hparams            = path.join('hparams', 'default.json')
    hparams            = JsonConfig(hparams)

    run_visualsearch(dataset_name, human_subject, trained_models_dir, hparams, device)