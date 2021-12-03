import torch
import torch.nn as nn
import torchvision.models as models
import json
from .. import constants
from . import utils
from torchvision import transforms
from os import listdir, path
from PIL import Image

"""
Runs the CNN feature extraction layers on each of the chopped stimuli images, alongside the target image.
The results are attention maps, which are then stored in the same folder as the chopped images, in JSON format.
"""

def run(trials_properties, targets_dir, chopped_dir):
    # Load the model
    model = models.vgg16(pretrained=True)
    print('Successfully loaded VGG16 model')
    """
    layers: num_layers, num_templates, conv_size
    layers: 5, 64, 14
    layers: 10, 128, 7
    layers: 17, 256, 4
    layers: 23, 512, 4
    layers: 24, 512, 2
    layers: 30, 512, 2
    layers: 31, 512, 1
    """
    conv_size     = 1
    num_layers    = 31
    num_templates = 512  

    model_target  = nn.Sequential(*list(model.features.children())[:num_layers])
    model_stimuli = nn.Sequential(*list(model.features.children())[:(num_layers - 1)])   

    # Set models in evaluation mode
    model_stimuli.eval()
    model_target.eval() 

    MMConv = nn.Conv2d(num_templates, 1, conv_size, padding=1)

    for index, trial in enumerate(trials_properties):
        image_name = trial['image']
        print('Working on ' + image_name + ' (' + str(index + 1) + '/' + str(len(trials_properties)) + ')...')  

        image_id = image_name[:-4]

        target_name = trial['target']
        target = Image.open(path.join(targets_dir, target_name)).convert('RGB')
        target_transformation = transforms.Compose([
            transforms.Resize((constants.CNN_TARGET_HEIGHT, constants.CNN_TARGET_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target       = target_transformation(target)
        # View as mini-batch of size 1
        # cast as 32-bit float since the model parameters are 32-bit floats
        batch_target = target.unsqueeze(0).float()
        # Get target representation and set as convolution layer's weight
        with torch.no_grad():
            output_target  = model_target(batch_target)
            MMConv.weight  = nn.parameter.Parameter(output_target, requires_grad=False)

        current_chopped_dir = path.join(chopped_dir, image_id)
        chopped_files       = listdir(current_chopped_dir)
        for chopped_image_name in chopped_files:
            # Check for precomputed files
            precomputed_file = path.join(current_chopped_dir, chopped_image_name[:-4] + '_layertopdown.json')
            if (chopped_image_name.endswith('.jpg') and path.exists(precomputed_file)) or chopped_image_name.endswith('_layertopdown.json'):
                continue    

            chopped_image = Image.open(path.join(current_chopped_dir, chopped_image_name)).convert('RGB')
            image_transformation = transforms.Compose([
                transforms.Resize((constants.CNN_IMAGE_HEIGHT, constants.CNN_IMAGE_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            chopped_image = image_transformation(chopped_image)

            batch_stimuli = chopped_image.unsqueeze(0).float()
            with torch.no_grad():
                # Get the feature map
                output_stimuli = model_stimuli(batch_stimuli).squeeze()
                # Output is the convolution of both representations
                chopped_attention_map = MMConv(output_stimuli.unsqueeze(0)).squeeze() 

            output    = {'x': chopped_attention_map.numpy().tolist()}
            save_file = path.join(current_chopped_dir, chopped_image_name[:-4] + '_layertopdown.json')

            utils.save_to_json(save_file, output)
