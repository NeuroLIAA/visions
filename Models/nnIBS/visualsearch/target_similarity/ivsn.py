import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from .target_similarity import TargetSimilarity
from torchvision import transforms
from skimage import io, img_as_ubyte, exposure, transform
from PIL import Image

""" Use attention map built by the IVSN model as a target similarity map. """
""" IVSN was created by authors Mengmi Zhang, Jiashi Feng, Keng Teck Ma, Joo Hwee Lim, Qi Zhao, and Gabriel Kreiman """
""" Full code, including a link to the paper, can be found at https://github.com/kreimanlab/VisualSearchZeroShot """
""" This code is an adaptation to PyTorch """

class Ivsn(TargetSimilarity):
    def compute_target_similarity(self, image, target, target_bbox):
        target_height, target_width = 32, 32
        block_height, block_width   = 224, 224
        # These blocks will be the input of the CNN
        image_blocks = self.divide_into_blocks(image, (block_height, block_width))

        # Load the model
        model = models.vgg16(pretrained=True)

        conv_size = 1
        num_layers = 31
        num_templates = 512  

        model_target = nn.Sequential(*list(model.features.children())[:num_layers])
        model_image  = nn.Sequential(*list(model.features.children())[:(num_layers - 1)])   

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
            # cast as 32-bit float since the model parameters are 32-bit floats
            batch_image  = block_data.unsqueeze(0).float()
            batch_target = target.unsqueeze(0).float()

            with torch.no_grad():
                # Get the feature maps
                output_stimuli = model_image(batch_image) .squeeze()
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

    def divide_into_blocks(self, image, block_size):
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