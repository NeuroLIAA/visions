import os
import json
import shutil

with open('./images_tasks.json', 'r') as fp:
    images_tasks = json.load(fp)

extension = '.pth.tar'
high_res_path = 'DCBs/HR/'
low_res_path  = 'DCBs/LR/'
for image in images_tasks.keys():
    if image[0] != '0':
        image_name = '0' + image[1:-4]
    else:
        image_name = image[:-4]

    new_name = image[:-4]
    image_info = images_tasks[image]
    task = image_info['task'].replace(' ', '_')
    shutil.move(high_res_path + task + '/' + image_name + extension, high_res_path + new_name + extension)
    shutil.move(low_res_path  + task + '/' + image_name + extension, low_res_path + new_name + extension)

# Clean up unused images
categories = [filename for filename in os.listdir(high_res_path) if os.path.isdir(high_res_path + filename)]
for category in categories:
    shutil.rmtree(high_res_path + category)
    shutil.rmtree(low_res_path + category)

