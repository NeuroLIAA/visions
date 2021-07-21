
""" Script which runs the IRL model on a given dataset
Usage:
  test.py <dataset_name>
  test.py -h | --help
Options:
  -h --help     Show this screen.
  --cuda=<id>   id of the cuda device [default: 0].
"""

import torch
import numpy as np
import json
from tqdm import tqdm
from docopt import docopt
from os import path, cpu_count
from dataset import process_eval_data
from irl_dcb.config import JsonConfig
from torch.utils.data import DataLoader
from irl_dcb.models import LHF_Policy_Cond_Small
from irl_dcb.environment import IRL_Env4LHF
from irl_dcb.build_belief_maps import build_belief_maps

from irl_dcb import utils

torch.manual_seed(42619)
np.random.seed(42619)

datasets_path = '../../Datasets/'
results_path  = '../../Results/'
dcbs_path = 'dataset_root/'

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
        progress = tqdm(test_img_loader)
        for i_batch, batch in enumerate(progress):
            env_test.set_data(batch)
            img_names_batch   = batch['img_name']
            cat_names_batch   = batch['cat_name']
            initial_fix_batch = batch['init_fix']
            with torch.no_grad():
                env_test.reset()
                trajs = utils.collect_trajs(env_test,
                                            generator,
                                            patch_num,
                                            max_traj_len,
                                            is_eval=True,
                                            sample_action=True)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i], initial_fix_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(env_test.batch_size)])
            
    scanpaths = utils.actions2scanpaths(all_actions, patch_num, im_w, im_h, dataset_name, hparams.Data.patch_size[0], max_traj_len)
    utils.cutFixOnTarget(scanpaths, bbox_annos)

    return scanpaths

if __name__ == '__main__':
    args = docopt(__doc__)
    device = torch.device('cpu')
    dataset_name = args["<dataset_name>"]
    checkpoint = 'trained_models/'
    hparams = path.join('hparams', 'default.json')
    hparams = JsonConfig(hparams)

    number_of_belief_maps = 134
    # Sigma of the gaussian filter applied to the search image
    sigma_blur = 2
    dcbs_path = path.join(dcbs_path, dataset_name)
    # Dir of pre-computed beliefs
    DCB_dir_HR = path.join(dcbs_path, 'DCBs/HR/')
    DCB_dir_LR = path.join(dcbs_path, 'DCBs/LR/')
    
    dataset_path = path.join(datasets_path, dataset_name)
    with open(path.join(dataset_path, 'trials_properties.json'), 'r') as json_file:
        trials_properties = json.load(json_file)
    
    with open(path.join(dataset_path, 'dataset_info.json'), 'r') as json_file:
        dataset_info = json.load(json_file)
    
    hparams.Data.max_traj_length = dataset_info['max_scanpath_length'] - 1
    images_dir = path.join(dataset_path, dataset_info['images_dir'])

    bbox_annos = {}
    iteration = 1
    for image_data in list(trials_properties):
        # If the target isn't categorized, remove it
        if image_data['target_object'] == 'TBD':
            trials_properties.remove(image_data)
            continue
            
        original_image_height = image_data['image_height']
        original_image_width  = image_data['image_width']
        # Rescale everything to image size used
        image_data['target_matched_column'] = utils.rescale_coordinate(image_data['target_matched_column'], original_image_width, hparams.Data.im_w)
        image_data['target_matched_row']    = utils.rescale_coordinate(image_data['target_matched_row'], original_image_height, hparams.Data.im_h)
        image_data['target_width']  = utils.rescale_coordinate(image_data['target_width'], original_image_width, hparams.Data.im_w)
        image_data['target_height'] = utils.rescale_coordinate(image_data['target_height'], original_image_height, hparams.Data.im_h)
        image_data['initial_fixation_column'] = utils.rescale_coordinate(image_data['initial_fixation_column'], original_image_width, hparams.Data.im_w)
        image_data['initial_fixation_row']    = utils.rescale_coordinate(image_data['initial_fixation_row'], original_image_height, hparams.Data.im_h)
        image_data['image_width']  = hparams.Data.im_w
        image_data['image_height'] = hparams.Data.im_h
        
        key = image_data['target_object'] + '_' + image_data['image']
        bbox_annos[key] = (image_data['target_matched_column'], image_data['target_matched_row'], image_data['target_width'], image_data['target_height'])

        # Create belief maps for image if necessary
        img_belief_maps_file = image_data['image'][:-4] + '.pth.tar'
        if not (path.exists(path.join(DCB_dir_HR, img_belief_maps_file)) and path.exists(path.join(DCB_dir_LR, img_belief_maps_file))):
            print('Building belief maps for image ' + image_data['image'] + ' (' + str(iteration) + '/' + str(len(trials_properties)) + ')')
            build_belief_maps(image_data['image'], 
                            images_dir,
                            (hparams.Data.im_w, hparams.Data.im_h),
                            (hparams.Data.patch_num[1], hparams.Data.patch_num[0]),
                            sigma_blur,
                            number_of_belief_maps,
                            DCB_dir_HR,
                            DCB_dir_LR)
        
        iteration += 1

    # Process fixation data
    dataset = process_eval_data(trials_properties,
                           DCB_dir_HR,
                           DCB_dir_LR,
                           bbox_annos,
                           hparams)
    
    batch_size = 64
    if len(trials_properties) < 64:
        batch_size = len(trials_properties)
    img_loader = DataLoader(dataset['img_test'],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=cpu_count())
    print('Number of images: ', len(dataset['img_test']))

    # Load trained model
    task_eye  = torch.eye(len(dataset['catIds'])).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(dataset['catIds']), task_eye,
                                      number_of_belief_maps).to(device)
    
    utils.load('best', generator, 'generator', pkg_dir=checkpoint, device=device)
    generator.eval()

    # Build environment
    env_test = IRL_Env4LHF(hparams.Data,
                           max_step=hparams.Data.max_traj_length,
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
                                hparams.Data.patch_num,
                                hparams.Data.max_traj_length,
                                hparams.Data.im_w,
                                hparams.Data.im_h,
                                num_sample=1)

    output_path = path.join(results_path, dataset_name + '_dataset' + '/IRL/')
    utils.save_scanpaths(output_path, predictions)