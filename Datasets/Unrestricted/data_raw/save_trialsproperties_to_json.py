from skimage import io, measure, color
from os import listdir, path
import utils

""" This script requires the gt files, which include the targets bboxes, to be located at '../gt' """

gt_dir   = '../gt'
gt_files = utils.sorted_alphanumeric(listdir(gt_dir))

targets_categories = utils.load_dict_from_json('targets_categories.json')

trials_properties = []
for gt in gt_files:
    img_id = gt[2:-4]
    gt_img = io.imread(path.join(gt_dir, gt))
    mask   = color.rgb2gray(gt_img) > 0.5
    # Label target region
    label_gt_img = measure.label(mask)
    # Get target region
    target = measure.regionprops(label_gt_img)
    start_row, start_column, end_row, end_column = target[0].bbox
    target_height = end_row - start_row - 1
    target_width  = end_column - start_column - 1

    img_height = gt_img.shape[0]
    img_width  = gt_img.shape[1]

    img_name = utils.get_name(int(img_id), 'image')
    tg_name  = utils.get_name(int(img_id), 'target')

    target_object = 'TBD'
    if img_name in targets_categories:
        target_object = targets_categories[img_name]['target_object']

    trials_properties.append({ "image" : img_name, "target" : tg_name, "dataset" : "Unrestricted Dataset", "target_matched_row" : start_row, "target_matched_column" : start_column, \
         "target_height" : target_height, "target_width" : target_width, "image_height" : img_height, "image_width" : img_width, \
         "initial_fixation_row" : 511, "initial_fixation_column" : 639, "target_object" : target_object})

utils.save_to_json('../trials_properties.json', trials_properties)