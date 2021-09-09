from .visibility_map  import VisibilityMap
from .visual_searcher import VisualSearcher
from .grid import Grid
from .utils import utils
from . import prior
import numpy as np
import time
import sys

" Runs the visual search model on the image/s specified with the supplied configuration "

def run(config, dataset_info, trials_properties, human_scanpaths, output_path, sigma):
    """ Input:
            Config (dict). One entry. Fields:
                search_model      (string)   : bayesian, greedy
                target_similarity (string)   : correlation, geisler
                prior             (string)   : deepgaze, mlnet, flat, center
                max_saccades      (int)      : maximum number of saccades allowed
                cell_size         (int)      : size (in pixels) of the cells in the grid
                scale_factor      (int)      : modulates the variance of target similarity and prevents 1 / d' from diverging in bayesian search
                additive_shift    (int)      : modulates the variance of target similarity and prevents 1 / d' from diverging in bayesian search
                save_probability_maps (bool) : indicates whether to save the probability map to a file after each saccade or not
                proc_number       (int)      : number of processes on which to execute bayesian search
                image_size        (int, int) : image size on which the model will operate
                save_similarity_maps (bool)  : indicates whether to save the target similarity map for each image in bayesian search
            Dataset info (dict). One entry. Fields:
                name          (string) : name of the dataset
                images_dir    (string) : folder path where search images are stored
                targets_dir   (string) : folder path where the targets are stored
                saliency_dir  (string) : folder path where the saliency maps are stored
                image_height  (int)    : default image height (in pixels)
                image_width   (int)    : default image width (in pixels)
            Trials properties (dict):
                Each entry specifies the data of the image on which to run the visual search model. Fields:
                image  (string)               : image name (where to look)
                target (string)               : name of the target image (what to look for)
                target_matched_row (int)      : starting Y coordinate, in pixels, of the target in the image
                target_matched_column (int)   : starting X coordinate, in pixels, of the target in the image
                target_height (int)           : height of the target in pixels
                target_width (int)            : width of the target in pixels
                initial_fixation_row (int)    : row of the first fixation on the image
                initial_fixation_column (int) : column of the first fixation on the image
            human_scanpaths (dict) : if not empty, it contains the human scanpaths which the model will use as fixations
            Output path (string)   : folder path where scanpaths and the probability maps will be stored
            sigma (2D array)       : covariance matrix used for building the visibility map
        Output:
            Output_path/scanpaths/Scanpaths.json: Dictionary indexed by image name where each entry contains the scanpath for that given image, alongside the configuration used.
            Output_path/probability_maps/: In this folder, the probability map computed for each saccade is stored. This is done for every image in trials_properties. (Only if save_probability_maps is true.)
            Output_path/similarity_maps/: In this folder, the target similarity map computed for each image is stored. This is done for every image in trials_properties. (Only if save_similarity_maps is true.)
    """
    images_dir    = dataset_info['images_dir']
    targets_dir   = dataset_info['targets_dir']
    saliency_dir  = dataset_info['saliency_dir']
    prior_name    = config['prior']
    
    image_size = (dataset_info['image_height'], dataset_info['image_width'])
    cell_size  = config['cell_size']

    model_image_size = config['image_size']
    
    # Initialize objects
    grid            = Grid(np.array(model_image_size), cell_size)
    visibility_map  = VisibilityMap(model_image_size, grid, sigma)
    visual_searcher = VisualSearcher(config, grid, visibility_map, output_path, human_scanpaths)

    # Rescale human scanpaths' coordinates (if any) to those of the grid
    utils.rescale_scanpaths(grid, human_scanpaths)

    print('Press Ctrl + C to interrupt execution and save a checkpoint \n')

    # If resuming execution, load previously generated data
    scanpaths, targets_found, previous_time = utils.load_data_from_checkpoint(output_path)

    trial_number = len(scanpaths.keys())
    total_trials = len(trials_properties) + trial_number
    start = time.time()
    try:
        for trial in trials_properties:
            trial_number += 1
            image_name  = trial['image']
            target_name = trial['target'] 
            print('Searching in image ' + image_name + ' (' + str(trial_number) + '/' + str(total_trials) + ')...')
            
            image       = utils.load_image(images_dir, image_name, model_image_size)
            target      = utils.load_image(targets_dir, target_name)
            image_prior = prior.load(image, image_name, model_image_size, prior_name, saliency_dir)
            
            initial_fixation = (trial['initial_fixation_row'], trial['initial_fixation_column'])
            initial_fixation = [utils.rescale_coordinate(initial_fixation[i], image_size[i], model_image_size[i]) for i in range(len(initial_fixation))]
            target_bbox      = [trial['target_matched_row'], trial['target_matched_column'], \
                                    trial['target_height'] + trial['target_matched_row'], trial['target_width'] + trial['target_matched_column']]
            target_bbox      = [utils.rescale_coordinate(target_bbox[i], image_size[i % 2 == 1], model_image_size[i % 2 == 1]) for i in range(len(target_bbox))]

            trial_scanpath = visual_searcher.search(image_name, image, image_prior, target, target_bbox, initial_fixation)

            if trial_scanpath:
                # If there were no errors, save the scanpath
                utils.add_scanpath_to_dict(image_name, trial_scanpath, target_bbox, trial['target_object'], grid, config, dataset_info['dataset_name'], scanpaths)
                if trial_scanpath['target_found']:
                    targets_found += 1
    except KeyboardInterrupt:
        time_elapsed = time.time() - start + previous_time
        utils.save_checkpoint(config, scanpaths, targets_found, trials_properties, time_elapsed, output_path)        
        sys.exit(0)

    time_elapsed = time.time() - start + previous_time
    breakpoint()
    utils.save_scanpaths(output_path, scanpaths, human_scanpaths)
    utils.erase_checkpoint(output_path)

    print('Total targets found: ' + str(targets_found) + '/' + str(len(scanpaths.keys())))
    print('Total time elapsed:  ' + str(round(time_elapsed, 4))   + ' seconds')