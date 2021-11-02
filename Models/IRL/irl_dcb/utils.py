import numpy as np
import torch
import json
import warnings
import pandas as pd
import os
from math import floor
from torch.distributions import Categorical
from Metrics.scripts import human_scanpath_prediction
warnings.filterwarnings("ignore", category=UserWarning)

def get_max_scanpath_length(scanpaths_list):
    if scanpaths_list:
        return max(list(map(lambda scanpath: len(scanpath['X']), scanpaths_list)))
    else:
        return 0

def rescale_coordinate(value, old_size, new_size):
    return floor((value / old_size) * new_size)

def are_within_boundaries(top_left_coordinates, bottom_right_coordinates, top_left_coordinates_to_compare, bottom_right_coordinates_to_compare):
    return top_left_coordinates[0] >= top_left_coordinates_to_compare[0] and top_left_coordinates[1] >= top_left_coordinates_to_compare[1] \
         and bottom_right_coordinates[0] < bottom_right_coordinates_to_compare[0] and bottom_right_coordinates[1] < bottom_right_coordinates_to_compare[1]

def add_scanpath_to_dict(model_name, image_name, image_size, scanpath_x, scanpath_y, target_object, patch_size, max_saccades, dataset_name, dict_):
    dict_[image_name] = {'subject' : model_name, 'dataset' : dataset_name + ' Dataset', 'image_height' : image_size[0], 'image_width' : image_size[1], \
        'receptive_height' : patch_size[1], 'receptive_width' : patch_size[0], 'target_found' : False, 'target_bbox' : np.zeros(shape=4), \
                 'X' : list(map(int, scanpath_x)), 'Y' : list(map(int, scanpath_y)), 'target_object' : target_object, 'max_fixations' : max_saccades + 1
        }

def probability_maps_for_batch(img_names_batch, output_path):
    exist_prob_maps_for_batch = True
    for image_name in img_names_batch:
        probability_maps_path = os.path.join(output_path, 'probability_maps', image_name[:-4])
        prob_maps_image = []
        if os.path.exists(probability_maps_path):
            prob_maps_image = os.listdir(probability_maps_path)

        if not prob_maps_image:
            exist_prob_maps_for_batch = False
            break
    
    return exist_prob_maps_for_batch

def save_and_compute_metrics(probs, human_scanpaths_batch, img_names_batch, output_path, presaved=False):
    for index, trial in enumerate(human_scanpaths_batch):
        trial_scanpath_x  = trial['X']
        trial_scanpath_y  = trial['Y']
        trial_target_bbox = trial['target_bbox']
        trial_length      = len(trial_scanpath_x)
        trial_img_name    = img_names_batch[index]

        # The initial fixation may have fallen under the target's bounding box due to grid rescaling
        initial_fixation = (trial_scanpath_x[0], trial_scanpath_y[0])
        if are_within_boundaries(initial_fixation, initial_fixation, (trial_target_bbox[0], trial_target_bbox[1]), (trial_target_bbox[2] + 1, trial_target_bbox[3] + 1)):
            # If this is the case, do not compute metrics
            continue

        save_path = os.path.join(output_path, 'probability_maps', trial_img_name[:-4])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not presaved:
            trial_prob_maps = [prob_batch[index] for prob_batch in probs[:trial_length - 1]]
            target_found_earlier = False
            for fix_number, prob_map in enumerate(trial_prob_maps):
                # The target may have been found earlier than in the subject's scanpath due to grid rescaling
                if target_found_earlier:
                    break
                current_fixation = (trial_scanpath_y[fix_number + 1], trial_scanpath_x[fix_number + 1])
                if are_within_boundaries(current_fixation, current_fixation, (trial_target_bbox[0], trial_target_bbox[1]), (trial_target_bbox[2] + 1, trial_target_bbox[3] + 1)):
                    target_found_earlier = True

                prob_map_df = pd.DataFrame(prob_map)
                prob_map_df.to_csv(os.path.join(save_path, 'fixation_' + str(fix_number + 1) + '.csv'), index=False)
        
        human_scanpath_prediction.save_scanpath_prediction_metrics(trial, trial_img_name, output_path)

def save_scanpaths(output_path, scanpaths, filename='Scanpaths.json'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_to_json(os.path.join(output_path, filename), scanpaths)

def save_to_json(file, data):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)

def cutFixOnTarget(trajs, target_annos, patch_size):
    targets_found = 0
    for image_name in trajs:
        traj = trajs[image_name]
        key = traj['target_object'] + '_' + image_name
        bbox = target_annos[key]
        traj_len = get_num_step2target(traj['X'], traj['Y'], bbox, patch_size)
        if traj_len != 1000:
            traj['target_found'] = True
            targets_found += 1
        traj['X'] = traj['X'][:traj_len]
        traj['Y'] = traj['Y'][:traj_len]
        traj['target_bbox'] = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]
    
    return targets_found

def pos_to_action(center_x, center_y, patch_size, patch_num):
    x = center_x // patch_size[0]
    y = center_y // patch_size[1]

    return int(patch_num[0] * y + x)

def action_to_pos(acts, patch_size, patch_num):
    patch_y = acts // patch_num[0]
    patch_x = acts % patch_num[0]

    pixel_x = patch_x * patch_size[0] + patch_size[0] / 2
    pixel_y = patch_y * patch_size[1] + patch_size[1] / 2
    return pixel_x, pixel_y

def select_action(obs, policy, sample_action, action_mask=None,
                  softmask=False, eps=1e-12):
    probs, values = policy(*obs)
    if sample_action:
        m = Categorical(probs)
        if action_mask is not None:
            # prevent sample previous actions by re-normalizing probs
            probs_new = probs.clone().detach()
            if softmask:
                probs_new = probs_new * action_mask
            else:
                probs_new[action_mask] = eps
            
            probs_new /= probs_new.sum(dim=1).view(probs_new.size(0), 1)
                
            m_new = Categorical(probs_new)
            actions = m_new.sample()

            probs = probs_new
        else:
            actions = m.sample()
        log_probs = m.log_prob(actions)
        return actions.view(-1), log_probs, values.view(-1), probs
    else:
        probs_new = probs.clone().detach()
        probs_new[action_mask.view(probs_new.size(0), -1)] = 0
        actions = torch.argmax(probs_new, dim=1)
        return actions.view(-1), None, None, None

def collect_trajs(env,
                  policy,
                  img_names_batch,
                  patch_num,
                  max_traj_length,
                  human_scanpaths_batch,
                  is_eval=True,
                  sample_action=True):
    obs_fov = env.observe()
    act, _, _, prob = select_action((obs_fov, env.task_ids),
                                               policy,
                                               sample_action,
                                               action_mask=env.action_mask)

    prob   = prob.view(prob.size(0), patch_num[1], -1)
    
    status = [env.status]
    probs  = [prob.numpy()]
    i = 0
    if is_eval:
        actions = []
        while i < max_traj_length:
            new_obs_fov, curr_status = env.step(act, human_scanpaths_batch)
            status.append(curr_status)
            actions.append(act)
            obs_fov = new_obs_fov
            act, _, _, prob = select_action(
                (obs_fov, env.task_ids),
                policy,
                sample_action,
                action_mask=env.action_mask)
            
            probs.append(prob.view(prob.size(0), patch_num[1], -1).numpy())
            i = i + 1

        trajs = {
            'status': torch.stack(status),
            'actions': torch.stack(actions)
        }

    return trajs, probs

def get_num_step2target(X, Y, bbox, patch_size):
    X, Y = np.array(X), np.array(Y)
    on_target_X = np.logical_and(X + patch_size[0] // 2 > bbox[0], X < bbox[0] + bbox[2] + patch_size[0] // 2)
    on_target_Y = np.logical_and(Y + patch_size[1] // 2 > bbox[1], Y < bbox[1] + bbox[3] + patch_size[1] // 2)
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    else:
        return 1000  # some big enough number

def get_num_steps(trajs, target_annos, task_names):
    num_steps = {}
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['name']
            bbox = target_annos[key]
            step_num = get_num_step2target(traj['X'], traj['Y'], bbox)
            num_steps_task[i] = step_num
            traj['X'] = traj['X'][:step_num]
            traj['Y'] = traj['Y'][:step_num]
        num_steps[task] = num_steps_task
    return num_steps

def calc_overlap_ratio(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    patch_area = float(patch_size[0] * patch_size[1])
    aoi_ratio = np.zeros((1, patch_num[1], patch_num[0]), dtype=np.float32)

    tl_x, tl_y = bbox[0], bbox[1]
    br_x, br_y = bbox[0] + bbox[2], bbox[1] + bbox[3]
    lx, ux = tl_x // patch_size[0], br_x // patch_size[0]
    ly, uy = tl_y // patch_size[1], br_y // patch_size[1]

    for x in range(lx, ux + 1):
        for y in range(ly, uy + 1):
            patch_tlx, patch_tly = x * patch_size[0], y * patch_size[1]
            patch_brx, patch_bry = patch_tlx + patch_size[
                0], patch_tly + patch_size[1]

            aoi_tlx = tl_x if patch_tlx < tl_x else patch_tlx
            aoi_tly = tl_y if patch_tly < tl_y else patch_tly
            aoi_brx = br_x if patch_brx > br_x else patch_brx
            aoi_bry = br_y if patch_bry > br_y else patch_bry

            aoi_ratio[0, y, x] = max((aoi_brx - aoi_tlx), 0) * max(
                (aoi_bry - aoi_tly), 0) / float(patch_area)

    return aoi_ratio

def foveal2mask(x, y, r, h, w):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist <= r
    return mask.astype(np.float32)

def multi_hot_coding(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    thresh = 0
    aoi_ratio = calc_overlap_ratio(bbox, patch_size, patch_num)
    hot_ind = aoi_ratio > thresh
    while hot_ind.sum() == 0:
        thresh *= 0.8
        hot_ind = aoi_ratio > thresh

    aoi_ratio[hot_ind] = 1
    aoi_ratio[np.logical_not(hot_ind)] = 0

    return aoi_ratio[0]

def actions2scanpaths(actions, patch_num, patch_size, im_w, im_h, dataset_name, max_saccades):
    scanpaths = {}
    for traj in actions:
        task_name, img_name, initial_fix, _, actions = traj
        actions = actions.to(dtype=torch.float32)
        py = (actions // patch_num[0]) / float(patch_num[1])
        px = (actions % patch_num[0]) / float(patch_num[0])
        fixs = torch.stack([px, py])
        fixs = np.concatenate([np.array([[float(initial_fix[0])], [float(initial_fix[1])]]),
                               fixs.cpu().numpy()],
                              axis=1)
        add_scanpath_to_dict('IRL Model', img_name, (im_h, im_w), fixs[0] * im_w, fixs[1] * im_h, task_name, patch_size, max_saccades, dataset_name, scanpaths)
    return scanpaths
                                      
def _file_best(name):
    return "trained_{}.pkg".format(name)

def load(step_or_path, model, name, optim=None, pkg_dir="", device=None):
    step = step_or_path
    save_path = None
    if isinstance(step, int):
        save_path = os.path.join(pkg_dir, _file_at_step(step, name))
    if isinstance(step, str):
        if pkg_dir is not None:
            if step == "best":
                save_path = os.path.join(pkg_dir, _file_best(name))
            else:
                save_path = os.path.join(pkg_dir, step)
        else:
            save_path = step
    if save_path is not None and not os.path.exists(save_path):
        print("[Checkpoint]: Failed to find {}".format(save_path))
        return
    if save_path is None:
        print("[Checkpoint]: Cannot load the checkpoint")
        return

    # begin to load
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    model.load_state_dict(state["model"])
    if optim is not None:
        optim.load_state_dict(state["optim"])

    print("Loaded {} successfully".format(save_path))
    return global_step
