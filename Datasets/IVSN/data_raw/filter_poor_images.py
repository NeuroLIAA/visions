import numpy as np
from os import listdir
import json

human_scanpaths_folder = '../human_scanpaths/'
poor_images = {}
for subject_scanpaths_file in listdir(human_scanpaths_folder):
    with open(human_scanpaths_folder + subject_scanpaths_file, 'r') as fp:
        subject_scanpaths = json.load(fp)
    for image_name in subject_scanpaths:
        if len(subject_scanpaths[image_name]['X']) < 2:
            if image_name in poor_images:
                poor_images[image_name] += 1
            else:
                poor_images[image_name] = 1
with open('short_scanpaths_per_image.json', 'w') as json_file:
    json.dump(poor_images, json_file, indent=4)
