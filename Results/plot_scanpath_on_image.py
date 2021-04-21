import json
from os import listdir, path, curdir
import math
from PIL import Image, ImageDraw

def plot_scanpath(scanpath_data,image_name):
    if scanpath_data['dataset'] == 'cIBS Dataset' or scanpath_data['dataset'] == 'cIBS':
        dataset_image_path = '../Datasets/cIBS/images/'
    elif scanpath_data['dataset'] == 'IVSN Natural Design' or scanpath_data['dataset'] == 'IVSN Natural Design Dataset':
        dataset_image_path = '../Datasets/IVSN/stimuli/' #estandaricemos los nombres que aparecen en el json
    with Image.open(dataset_image_path + image_name) as im:
        target_size = (200,200) #acá iría el tamaño del target que no está en el json
        target_center = (400,400) #acá irían las coordenadas del target que no están en el json
        half_window_size = tuple(map(lambda i: i / 2, target_size))
        target_top_left_coordinates = tuple(map(lambda i, j: i - j, target_center, half_window_size))
        target_bottom_right_coordinates = tuple(map(lambda i, j: i + j, target_center, half_window_size))
        target_shape = [target_top_left_coordinates, target_bottom_right_coordinates] 
        
        image_with_scanpath_plotted = ImageDraw.Draw(im)  
        image_with_scanpath_plotted.rectangle(target_shape, fill ='#0000ff2f', outline ="green")
        fixations = list(zip(scanpath_data['X'], scanpath_data['Y']))
        fixation_window_size = (32,32) #acá iría el delta que no está en el json
        for first_fixation_index in range(len(fixations)):
            half_window_size = tuple(map(lambda i: i / 2, fixation_window_size))
            fixation_top_left_coordinates = tuple(map(lambda i, j: i - j, fixations[first_fixation_index], half_window_size))
            fixation_bottom_right_coordinates = tuple(map(lambda i, j: i + j, fixations[first_fixation_index], half_window_size))
            fixation_shape = [fixation_top_left_coordinates, fixation_bottom_right_coordinates]
            if len(fixations) > first_fixation_index+1:
                saccade_shape = (fixations[first_fixation_index], fixations[first_fixation_index+1])
                image_with_scanpath_plotted.line(saccade_shape, fill ='#00ff002f', width = 3)
            image_with_scanpath_plotted.rectangle(fixation_shape, fill ='#ff00002f', outline ="red")
        im.show()


def pick_three_scanpaths_to_plot(scanpaths):
    i = 0
    for image_name in scanpaths.keys():
        i += 1
        plot_scanpath(scanpaths[image_name],image_name)
        if i ==3:
            break
            
results_dir  =  curdir + '/'
dataset_results_dirs = listdir(results_dir)
for dataset in dataset_results_dirs:
    if not(path.isdir(path.join(results_dir, dataset))):
        continue
    dataset_name = dataset.split('_')[0]
    dataset_results_dir = results_dir + dataset + '/'
    models = listdir(dataset_results_dir)
    for model_name in models:
        if not(path.isdir(path.join(dataset_results_dir, model_name))):
            continue

        model_scanpaths_file = dataset_results_dir + model_name + '/Scanpaths.json'
        with open(model_scanpaths_file, 'r') as fp:
            model_scanpaths = json.load(fp)        
            pick_three_scanpaths_to_plot(model_scanpaths)



