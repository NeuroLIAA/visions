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