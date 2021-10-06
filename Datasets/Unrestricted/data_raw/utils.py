import json
import re

def between_bounds(target_bbox, fix_y, fix_x, receptive_size):
    return target_bbox[0] <= fix_y + receptive_size[0] // 2 and target_bbox[2] >= fix_y - receptive_size[0] // 2 and \
            target_bbox[1] <= fix_x + receptive_size[1] // 2 and target_bbox[3] >= fix_x - receptive_size[1] // 2

def crop_scanpath(scanpath_x, scanpath_y, target_bbox, receptive_size):
    target_found = False
    cropped_scanpath_x = list(scanpath_x)
    cropped_scanpath_y = list(scanpath_y)
    for index, fixation in enumerate(zip(scanpath_y, scanpath_x)):
        if between_bounds(target_bbox, fixation[0], fixation[1], receptive_size):
            cropped_scanpath_x = scanpath_x[:index + 1]
            cropped_scanpath_y = scanpath_y[:index + 1]
            target_found = True
            break
    
    return target_found, cropped_scanpath_x, cropped_scanpath_y

def collapse_fixations(scanpath_x, scanpath_y, receptive_size):
    collapsed_scanpath_x = list(scanpath_x)
    collapsed_scanpath_y = list(scanpath_y)
    index = 0
    while index < len(collapsed_scanpath_x) - 1:
        abs_difference_x = [abs(fix_1 - fix_2) for fix_1, fix_2 in zip(collapsed_scanpath_x, collapsed_scanpath_x[1:])]
        abs_difference_y = [abs(fix_1 - fix_2) for fix_1, fix_2 in zip(collapsed_scanpath_y, collapsed_scanpath_y[1:])]

        if abs_difference_x[index] < receptive_size[1] / 2 and abs_difference_y[index] < receptive_size[0] / 2:
            new_fix_x = (collapsed_scanpath_x[index] + collapsed_scanpath_x[index + 1]) / 2
            new_fix_y = (collapsed_scanpath_y[index] + collapsed_scanpath_y[index + 1]) / 2
            collapsed_scanpath_x[index] = new_fix_x
            collapsed_scanpath_y[index] = new_fix_y
            del collapsed_scanpath_x[index + 1]
            del collapsed_scanpath_y[index + 1]
        else:
            index += 1

    return collapsed_scanpath_x, collapsed_scanpath_y

def get_trial_for_image_name(trials_properties, image_name):
    for trial in trials_properties:
        if trial['image'] == image_name:
            return trial

def get_name(img_id, _type):
    str_id = str(img_id)
    if img_id < 100:
        str_id = '0' + str_id
    if img_id < 10:
        str_id = '0' + str_id

    if (_type == 'image'):
        name = 'img' + str_id + '.jpg'
    else:
        name = 't' + str_id + '.jpg'

    return name

def load_filtered_images():
    filtered_images_file = open('trivial_images.txt', 'r')
    filtered_images = []
    for line in filtered_images_file:
        filtered_images.append(line.strip())
    
    filtered_images_file.close()
    return filtered_images

def load_dict_from_json(json_file):
    with open(json_file, 'r') as fp:
        return json.load(fp)

def save_to_json(json_file, data):
    with open(json_file, 'w') as fp:
        json.dump(data, fp, indent=4)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def rescale_coordinate(value, old_size, new_size):
    return (value / old_size) * new_size