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