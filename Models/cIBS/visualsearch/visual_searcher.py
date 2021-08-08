from .models.bayesian_model import BayesianModel
from .models.greedy_model   import GreedyModel
from .utils import utils
from . import prior
import numpy as np
import time
import importlib

class VisualSearcher: 
    def __init__(self, config, grid, visibility_map, output_path):
        " Creates a new instance of the visual search model "
        """ Input:
                Config (dict). One entry. Fields:
                    search_model          (string) : bayesian, greedy
                    target_similarity     (string) : correlation, geisler
                    prior                 (string) : deepgaze, mlnet, flat, center
                    max_saccades          (int)    : maximum number of saccades allowed
                    cell_size             (int)    : size (in pixels) of the cells in the grid
                    scale_factor          (int)    : modulates the variance of target similarity and prevents 1 / d' from diverging in bayesian search
                    additive_shift        (int)    : modulates the variance of target similarity and prevents 1 / d' from diverging in bayesian search
                    save_probability_maps (bool)   : indicates whether to save the posterior to a file after each saccade or not
                    proc_number           (int)    : number of processes on which to execute bayesian search
                grid           (Grid)          : representation of an image with cells instead of pixels
                visibility_map (VisibilityMap) : visibility map with the size of the grid
                Output path    (string)        : folder path where scanpaths and probability maps will be stored
        """
        self.max_saccades             = config['max_saccades']
        self.grid                     = grid
        self.scale_factor             = config['scale_factor']
        self.additive_shift           = config['additive_shift']
        self.seed                     = config['seed']
        self.save_posterior           = config['save_probability_maps']
        self.visibility_map           = visibility_map
        self.search_model             = self.initialize_model(config['search_model'], self.grid.size(), visibility_map, config['norm_cdf_tolerance'], config['proc_number'])
        self.target_similarity_method = config['target_similarity']
        self.output_path              = output_path        

    def search(self, image_name, image, image_prior, target, target_bbox, initial_fixation):
        " Given an image, a target, and a prior of that image, it looks for the object in the image, generating a scanpath "
        """ Input:
            Specifies the data of the image on which to run the visual search model. Fields:
                image_name (string)         : name of the image
                image (2D array)            : search image
                image_prior (2D array)      : grayscale image with values between 0 and 1 that serves as prior
                target (2D array)           : target image
                target_bbox (array)         : bounding box (upper left row, upper left column, lower right row, lower right column) of the target inside the search image
                initial_fixation (int, int) : row and column of the first fixation on the search image
            Output:
                image_scanpath   (dict)      : scanpath made by the model on the search image, alongside a 'target_found' field which indicates if the target was found
                probability_maps (csv files) : if self.save_posterior is True, the posterior of each saccade is stored in a .csv file inside a folder in self.output_path 
        """
        # Convert prior to grid
        image_prior = self.grid.reduce(image_prior, mode='mean')
        grid_size   = self.grid.size()
        # Check prior dimensions
        if not(image_prior.shape == grid_size):
            print(image_name + ': prior image\'s dimensions don\'t match dataset\'s dimensions')
            return {}
        # Sum probabilities
        image_prior = prior.sum(image_prior, self.max_saccades)
      
        # Convert target bounding box to grid cells
        target_bbox_in_grid = np.empty(len(target_bbox), dtype=np.int)
        target_bbox_in_grid[0], target_bbox_in_grid[1] = self.grid.map_to_cell((target_bbox[0], target_bbox[1]))
        target_bbox_in_grid[2], target_bbox_in_grid[3] = self.grid.map_to_cell((target_bbox[2], target_bbox[3]))
        if not(utils.are_within_boundaries((target_bbox_in_grid[0], target_bbox_in_grid[1]), (target_bbox_in_grid[2], target_bbox_in_grid[3]), np.zeros(2), grid_size)):
            print(image_name + ': target bounding box is outside of the grid')
            return {}
        
        # Initialize fixations matrix
        fixations    = np.empty(shape=(self.max_saccades + 1, 2), dtype=int)
        fixations[0] = self.grid.map_to_cell(initial_fixation)
        if not(utils.are_within_boundaries(fixations[0], fixations[0], np.zeros(2), grid_size)):
            print(image_name + ': initial fixation falls off the grid')
            return {}

        target_similarity_map = self.initialize_target_similarity_map(image, target, target_bbox)

        # Initialize variables for computing each fixation        
        likelihood = np.zeros(shape=grid_size)
        posterior  = np.zeros(shape=grid_size)

        # Search
        print('Fixation:', end=' ')
        target_found = False
        start = time.time()
        for fixation_number in range(self.max_saccades + 1):
            current_fixation = fixations[fixation_number]
            print(fixation_number + 1, end=' ')
            
            if utils.are_within_boundaries(current_fixation, current_fixation, (target_bbox_in_grid[0], target_bbox_in_grid[1]), (target_bbox_in_grid[2] + 1, target_bbox_in_grid[3] + 1)):
                target_found = True
                fixations = fixations[:fixation_number + 1]
                break

            # If the limit has been reached, don't compute the next fixation
            if fixation_number == self.max_saccades:
                break

            if fixation_number == 0:
                likelihood = target_similarity_map.at_fixation(current_fixation) * (np.square(self.visibility_map.at_fixation(current_fixation)))
                likelihood_times_prior = image_prior * np.exp(likelihood)
            else:
                likelihood = likelihood + target_similarity_map.at_fixation(current_fixation) * (np.square(self.visibility_map.at_fixation(current_fixation)))
                likelihood_times_prior = posterior * np.exp(likelihood)

            marginal  = np.sum(likelihood_times_prior)
            posterior = likelihood_times_prior / marginal

            if self.save_posterior:
                utils.save_probability_map(self.output_path, image_name, posterior, fixation_number)

            fixations[fixation_number + 1] = self.search_model.next_fixation(posterior)
            
        end = time.time()

        if target_found:
            print('\nTarget found!')
        else:
            print('\nTarget NOT FOUND!')
        print('Time elapsed: ' + str(end - start) + '\n')

        # Revert back to pixels
        # fixations = [self.grid.map_cell_to_pixels(fixation) for fixation in fixations]

        # Note: each x coordinate refers to a column in the image, and each y coordinate refers to a row in the image
        scanpath_x_coordinates = self.get_coordinates(fixations, axis=1)
        scanpath_y_coordinates = self.get_coordinates(fixations, axis=0)

        return { 'target_found' : target_found, 'scanpath_x' : scanpath_x_coordinates, 'scanpath_y' : scanpath_y_coordinates }
    
    def get_coordinates(self, fixations, axis):
        fixations_as_list = np.array(fixations).flatten()

        return [fixations_as_list[fix_number] for fix_number in range(axis, len(fixations_as_list), 2)]

    def initialize_model(self, search_model, grid_size, visibility_map, norm_cdf_tolerance, number_of_processes):
        if search_model == 'greedy':
            return GreedyModel()
        else:
            return BayesianModel(grid_size, visibility_map, norm_cdf_tolerance, number_of_processes)

    def initialize_target_similarity_map(self, image, target, target_bbox):
        # Load corresponding module
        module = importlib.import_module('.target_similarity.' + self.target_similarity_method.lower(), 'visualsearch')
        # Get the class
        target_similarity_class = getattr(module, self.target_similarity_method.capitalize())
        target_similarity_map   = target_similarity_class(image, target, target_bbox, self.visibility_map, self.scale_factor, self.additive_shift, self.grid, self.seed)
        return target_similarity_map
