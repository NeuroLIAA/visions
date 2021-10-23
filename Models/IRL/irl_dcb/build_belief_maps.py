import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter
from .run_detectron import run_detectron
from os import path, makedirs

def create_categories_masks(number_of_categories, output_size, segments_info, panoptic_seg):
    belief_maps = np.zeros(shape=(number_of_categories, output_size[0], output_size[1]), dtype=np.int32)
    for segment in segments_info:
        category_id   = segment['category_id']
        category_mask = (panoptic_seg == segment['id']) * 1
        
        if not segment['isthing']:
            category_id += 80
        
        belief_maps[category_id] += category_mask    
    
    return belief_maps

def build_belief_maps(image_name, images_dir, image_size, output_size, sigma_blur, number_of_categories, save_path_HR, save_path_LR):
    """ Build belief maps for a given image """
    """ Image size is in width x height """
    image = cv2.imread(path.join(images_dir, image_name))
    image = cv2.resize(image, image_size)
    blurred_image = gaussian_filter(image, (sigma_blur, sigma_blur, 0), mode='nearest')

    # hi_res_panoptic_seg, hi_res_segments_info   = run_detectron(image, output_shape=output_size)
    # low_res_panoptic_seg, low_res_segments_info = run_detectron(blurred_image, output_shape=output_size)

    hi_res_panoptic_seg, hi_res_segments_info   = run_detectron(image)
    low_res_panoptic_seg, low_res_segments_info = run_detectron(blurred_image)

    # Rescale to output_size
    hi_res_panoptic_seg  = hi_res_panoptic_seg[np.newaxis, np.newaxis, :, :]
    hi_res_panoptic_seg  = torch.nn.functional.interpolate(hi_res_panoptic_seg.float(), size=output_size)
    low_res_panoptic_seg = low_res_panoptic_seg[np.newaxis, np.newaxis, :, :]
    low_res_panoptic_seg = torch.nn.functional.interpolate(low_res_panoptic_seg.float(), size=output_size)

    belief_maps_hi_res  = torch.from_numpy(create_categories_masks(number_of_categories, output_size, hi_res_segments_info, hi_res_panoptic_seg.numpy()[0, 0, :, :]))
    belief_maps_low_res = torch.from_numpy(create_categories_masks(number_of_categories, output_size, low_res_segments_info, low_res_panoptic_seg.numpy()[0, 0, :, :]))

    filename = image_name[:-4] + '.pth.tar'

    if not path.exists(save_path_HR):
        makedirs(save_path_HR)
    if not path.exists(save_path_LR):
        makedirs(save_path_LR)
    
    torch.save(belief_maps_hi_res, path.join(save_path_HR, filename))
    torch.save(belief_maps_low_res, path.join(save_path_LR, filename))