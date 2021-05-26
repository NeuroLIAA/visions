import json
from skimage import io, transform, img_as_ubyte
from os import listdir, path, curdir
import math
from PIL import Image, ImageDraw, ImageFont

def plot_scanpath(scanpath_data,image_name):
    if scanpath_data['dataset'] == 'cIBS Dataset' or scanpath_data['dataset'] == 'cIBS':
        dataset_image_path = '../Datasets/cIBS/images/'
    elif scanpath_data['dataset'] == 'IVSN Natural Design' or scanpath_data['dataset'] == 'IVSN Natural Design Dataset':
        dataset_image_path = '../Datasets/IVSN/stimuli/' #estandaricemos los nombres que aparecen en el json
 
        im = io.imread(dataset_image_path + image_name)
        im = img_as_ubyte(transform.resize(im, (scanpath_data['image_height'],scanpath_data['image_width'])))
        im = Image.fromarray(im)
        im = im.convert("RGBA")
        target_bbox = scanpath_data['target_bbox']
        target_top_left_coordinates = (target_bbox[1],target_bbox[0])
        target_bottom_right_coordinates = (target_bbox[3],target_bbox[2])
        target_shape = [target_top_left_coordinates, target_bottom_right_coordinates] 
        
        image_with_scanpath_plotted = ImageDraw.Draw(im)
        fixation_order_font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 16)
        image_with_scanpath_plotted.rectangle(target_shape, fill ='#0000ffff', outline ="green")
        fixations = list(zip(scanpath_data['X'], scanpath_data['Y']))
        fixation_window_size = (scanpath_data['receptive_width'],scanpath_data['receptive_height'])
        fixation_shapes = []
        for first_fixation_index in range(len(fixations)): #primero agrego las lineas
            half_window_size = tuple(map(lambda i: i / 2, fixation_window_size))
            fixation_top_left_coordinates = tuple(map(lambda i, j: i - j, fixations[first_fixation_index], half_window_size))
            fixation_bottom_right_coordinates = tuple(map(lambda i, j: i + j, fixations[first_fixation_index], half_window_size))
            fixation_shape = [fixation_top_left_coordinates, fixation_bottom_right_coordinates]
            if len(fixations) > first_fixation_index+1:
                saccade_shape = (fixations[first_fixation_index], fixations[first_fixation_index+1])
                image_with_scanpath_plotted.line(saccade_shape, fill ='#00ff00ff', width = 3)
            fixation_shapes.append(fixation_shape)   
        for fixation_shape in fixation_shapes: #segundo agrego los cuadrados
            image_with_scanpath_plotted.rectangle(fixation_shape, fill ='#ff0000ff', outline ="red")
        image_with_scanpath_plotted.rectangle(target_shape, fill ='#0000ffff', outline ="green")
        for fixation_shape in fixation_shapes: #por último agrego los números. Todo esto para que las lineas y los cuadrados no sean tapados
            image_with_scanpath_plotted.text(fixations[fixation_shapes.index(fixation_shape)], str(fixation_shapes.index(fixation_shape)+1), font=fixation_order_font, fill='#000000ff')
        
        im.save(image_name[0:-4] + ".png")
        im.show()
	
def pick_three_scanpaths_to_plot(scanpaths):
   # i = 0
    for image_name in scanpaths.keys():
        #i += 1
        struct = scanpaths[image_name]
        if struct['target_found'] == False and struct['dataset'] == 'IVSN Natural Design Dataset' and struct['subject'] == 'cIBS model':
        	plot_scanpath(scanpaths[image_name],image_name)
        #if i ==3:
         #   break
            
results_dir  = '../Results/'
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



