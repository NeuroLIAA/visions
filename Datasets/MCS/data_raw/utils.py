import json

def load_dict_from_json(json_file):
    with open(json_file, 'r') as fp:
        return json.load(fp)

def save_to_json(json_file, data):
    with open(json_file, 'w') as fp:
        json.dump(data, fp, indent=4)

def between_bounds(target_bbox, fix_y, fix_x, receptive_size):
    return target_bbox[0] <= fix_y + receptive_size[0] // 2 and target_bbox[2] >= fix_y - receptive_size[0] // 2 and \
            target_bbox[1] <= fix_x + receptive_size[1] // 2 and target_bbox[3] >= fix_x - receptive_size[1] // 2

def crop_scanpath(scanpath_x, scanpath_y, target_bbox, receptive_size):
    target_found = False
    cropped      = False
    original_len = len(scanpath_x)
    for index, fixation in enumerate(zip(scanpath_y, scanpath_x)):
        if between_bounds(target_bbox, fixation[0], fixation[1], receptive_size):
            scanpath_x = scanpath_x[:index + 1]
            scanpath_y = scanpath_y[:index + 1]
            target_found = True
            break
    
    if len(scanpath_x) < original_len: cropped = True
    
    return target_found, cropped

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

def convert_coordinate(X, Y, im_w, im_h, display_size, is_train):
    """
    convert from display coordinate to pixel coordinate

    X - x coordinate of the fixations
    Y - y coordinate of the fixations
    im_w - image width
    im_h - image height
    is_train - is the fixations drawn from training data
    """
    if is_train:
        display_h, display_w = display_size
        dif_ux = int((display_w - im_w) / 2)
        dif_uy = int((display_h - im_h) / 2)
        X = X - dif_ux
        Y = Y - dif_uy
    else:
        display_h, display_w = display_size
        target_ratio = display_w / float(display_h)
        ratio = im_w / float(im_h)

        delta_w, delta_h = 0, 0
        if ratio > target_ratio:
            new_w = display_w
            new_h = int(new_w / ratio)
            delta_h = display_h - new_h
        else:
            new_h = display_h
            new_w = int(new_h * ratio)
            delta_w = display_w - new_w
        dif_ux = delta_w // 2
        dif_uy = delta_h // 2
        scale = im_w / float(new_w)
        X = (X - dif_ux) * scale
        Y = (Y - dif_uy) * scale
    return X, Y

def rescale_coordinate(value, old_size, new_size):
    return (value / old_size) * new_size

def rename_image(image_name, category):
    for zeros in range(12 - len(image_name[:-4])):
        image_name = '0' + image_name
    
    if category == 'microwave':
        image_name = 'm' + image_name
    else:
        image_name = 'c' + image_name

    return image_name