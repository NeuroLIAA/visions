import torch
import torch.nn as nn
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
import numpy as np
from torchvision import transforms
from skimage import img_as_ubyte, exposure, transform
from PIL import Image

def compute_target_similarity(image, target):
    target_height, target_width = 128, 128
    block_height, block_width   = 224, 224
    # These blocks will be the input of the CNN
    image_blocks = divide_into_blocks(image, (block_height, block_width))

    # Load the model
    model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

    conv_size = 1
    num_templates = 2048

    model_image  = nn.Sequential(*list(model.children())[:-2])
    model_target = nn.Sequential(model_image, nn.AdaptiveMaxPool2d(output_size=1))

    # Set models in evaluation mode
    model_image.eval()
    model_target.eval() 

    MMConv = nn.Conv2d(num_templates, 1, conv_size, padding=1)

    target = Image.fromarray(target).convert('RGB')
    target_transformation = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target = target_transformation(target)

    target_similarity_map = np.zeros(shape=image.shape[:2])
    for image_block in image_blocks:
        block_data = Image.fromarray(image_block['img_block']).convert('RGB')
        image_transformation = transforms.Compose([
            transforms.Resize((block_height, block_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        block_data = image_transformation(block_data)

        # View as mini-batch of size 1
        batch_image  = block_data.unsqueeze(0)
        batch_target = target.unsqueeze(0)

        with torch.no_grad():
            # Get the feature maps
            output_stimuli = model_image(batch_image).squeeze()
            output_target  = model_target(batch_target)
            MMConv.weight = nn.parameter.Parameter(output_target, requires_grad=False)
            # Output is the convolution of both representations
            out = MMConv(output_stimuli.unsqueeze(0)).squeeze()

        target_similarity_block = transform.resize(out.numpy(), image_block['img_block'].shape[:2])
        from_row    = image_block['from_row']
        from_column = image_block['from_column']
        to_row    = from_row + block_height
        to_column = from_column + block_width
        target_similarity_map[from_row:to_row, from_column:to_column] = target_similarity_block

    target_similarity_map = exposure.rescale_intensity(target_similarity_map, out_range=(0, 1))
    return target_similarity_map

def divide_into_blocks(image, block_size):
    image_blocks = []
    img_height, img_width = image.shape[0], image.shape[1]
    default_block_height, default_block_width = block_size

    number_of_rows    = img_height // default_block_height
    number_of_columns = img_width // default_block_width
    extra_row    = (img_height % default_block_height > 0)
    extra_column = (img_width % default_block_width > 0)
    if (extra_row):
        extra_row_height = img_height % default_block_height
        number_of_rows += 1
    if (extra_column):
        extra_column_width = img_width % default_block_width
        number_of_columns += 1
    
    for row in range(number_of_rows):
        current_block_size = (default_block_height, default_block_width)
        if (extra_row and (row + 1) == number_of_rows):
            current_block_size = (extra_row_height, current_block_size[1])
        for column in range(number_of_columns):            
            if (extra_column and (column + 1) == number_of_columns):
                current_block_size = (current_block_size[0], extra_column_width)
            from_row    = row * default_block_height
            from_column = column * default_block_width
            to_row    = from_row + current_block_size[0]
            to_column = from_column + current_block_size[1]
            img_crop = image[from_row:to_row, from_column:to_column]

            image_blocks.append({'from_row' : from_row, 'from_column' : from_column, 'img_block' : img_as_ubyte(img_crop)})
    
    return image_blocks