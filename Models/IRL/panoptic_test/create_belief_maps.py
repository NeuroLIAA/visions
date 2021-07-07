import json
import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter
from run_detectron import run_detectron
from os import listdir, path, makedirs

images_dir = '../../../Datasets/COCOSearch18/images/'
images_files = listdir(images_dir)
hi_res_save_path  = '../Cocosearch18_trainval/DCBs_new/HR/'
low_res_save_path = '../Cocosearch18_trainval/DCBs_new/LR/'

# 2 degrees of vision
sigma_blur = 2

number_of_categories = 134
output_size = (20, 32)

def create_categories_masks(number_of_categories, output_size, segments_info, panoptic_seg):
    belief_maps = np.zeros(shape=(number_of_categories, output_size[0], output_size[1]), dtype=np.int32)
    for segment in segments_info:
        category_id   = segment['category_id']
        category_mask = (panoptic_seg == segment['id']) * 1
        
        if not segment['isthing']:
            category_id += 80
        
        belief_maps[category_id] += category_mask    
    
    return belief_maps

for image_file in images_files:
    if not image_file.endswith('.jpg'):
        continue
    image = cv2.imread(path.join(images_dir, image_file))
    image = cv2.resize(image, (512, 320))
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

    filename = image_file[:-4] + '.pth.tar'

    if not path.exists(hi_res_save_path):
         makedirs(hi_res_save_path)
    if not path.exists(low_res_save_path):
        makedirs(low_res_save_path)
    
    torch.save(belief_maps_hi_res, path.join(hi_res_save_path, filename))
    torch.save(belief_maps_low_res, path.join(low_res_save_path, filename))