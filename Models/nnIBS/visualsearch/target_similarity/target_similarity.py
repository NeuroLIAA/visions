import numpy as np
from os import path
from skimage import io
from ..utils import utils

class TargetSimilarity():
    def __init__(self, image_name, image, target, target_bbox, visibility_map, scale_factor, additive_shift, grid, seed, number_of_processes, save_similarity_maps, target_similarity_dir):
        # Set the seed for generating random noise
        np.random.seed(seed)

        self.number_of_processes   = number_of_processes
        self.save_similarity_maps  = save_similarity_maps
        self.grid                  = grid
        self.image_name            = image_name
        self.target_similarity_dir = target_similarity_dir

        self.create_target_similarity_map(image, target, target_bbox, visibility_map, scale_factor, additive_shift)

    def create_target_similarity_map(self, image, target, target_bbox, visibility_map, scale_factor, additive_shift):
        " Creates the target similarity map for a given image, target and visibility map.  "
        """ Input:
                image  (2D array) : search image
                target (2D array) : target image
                target_bbox (array)  : bounding box (upper left row, upper left column, lower right row, lower right column) of the target in the image
                visibility_map (VisibilityMap) : visibility map which indicates how focus decays over distance from the fovea
                scale_factor   (int) : modulates the inverse of the visibility and prevents the variance from diverging
                additive_shift (int) : modulates the inverse of the visibility and prevents the variance from diverging
            Output:
                sigma, mu (4D arrays) : values of the normal distribution for each possible fixation in the grid. It's based on target similarity and visibility
        """
        grid_size = self.grid.size()
        target_bbox_in_grid = np.empty(len(target_bbox), dtype=np.int)
        target_bbox_in_grid[0], target_bbox_in_grid[1] = self.grid.map_to_cell((target_bbox[0], target_bbox[1]))
        target_bbox_in_grid[2], target_bbox_in_grid[3] = self.grid.map_to_cell((target_bbox[2], target_bbox[3]))

        # Initialize mu, where each cell has a value of 0.5 if the target is present and -0.5 otherwise
        self.mu = np.zeros(shape=(grid_size[0], grid_size[1], grid_size[0], grid_size[1])) - 0.5
        for row in range(target_bbox_in_grid[0], target_bbox_in_grid[2] + 1):
            for column in range(target_bbox_in_grid[1], target_bbox_in_grid[3] + 1):
                self.mu[row, column] = np.zeros(shape=grid_size) + 0.5
        
        # Initialize sigma
        self.sigma = np.ones(shape=self.mu.shape)
        # Variance now depends on the visibility
        self.sigma = self.sigma / (visibility_map.normalized_at_every_fixation() * scale_factor + additive_shift)
              
        # If precomputed, load target similarity map
        save_path = path.join(self.target_similarity_dir, self.__class__.__name__)
        filename  = self.image_name[:-4] + '.png'
        file_path = path.join(save_path, filename)
        if path.exists(file_path):
            target_similarity_map = io.imread(file_path)
        else:
            # Calculate target similarity based on a specific method  
            print('Building target similarity map...')
            target_similarity_map = self.compute_target_similarity(image, target, target_bbox)
            if self.save_similarity_maps:
                utils.save_similarity_map(save_path, filename, target_similarity_map)
        
        # Add target similarity and visibility info to mu
        self.add_info_to_mu(target_similarity_map, visibility_map)

        return

    def compute_target_similarity(self, image, target, target_bbox):
        """ Each subclass calculates the target similarity map with its own method """
        pass

    def add_info_to_mu(self, target_similarity_map, visibility_map):
        """ Once target similarity has been computed, its information is added to mu, alongside the visibility map """
        # Reduce to grid
        target_similarity_map = self.grid.reduce(target_similarity_map, mode='max')

        # Convert values to the interval [-0.5, 0.5] 
        target_similarity_map = target_similarity_map - np.min(target_similarity_map)
        target_similarity_map = target_similarity_map / np.max(target_similarity_map) - 0.5
        # Make it the same shape as mu
        grid_size = self.grid.size()
        target_similarity_map = np.tile(target_similarity_map[:, :, np.newaxis, np.newaxis], (1, 1, grid_size[0], grid_size[1]))

        # Modify mu in order to incorporate target similarity and visibility
        self.mu = self.mu * (visibility_map.normalized_at_every_fixation() + 0.5) + target_similarity_map * (1 - visibility_map.normalized_at_every_fixation() + 0.5)
        # Convert values to the interval [-1, 1]
        self.mu = self.mu / 2

        return
    
    def at_fixation(self, fixation):
        " Given a fixation in the grid, it returns the target similarity map, represented as a 2D array of scalars with added random noise "
        """ Input:
                fixation (int, int) : cell in the grid on which the observer is fixating
            Output:
                target_similarity_map (2D array of floats) : matrix the size of the grid, where each value is a scalar which represents how similar the position is to the target
        """
        grid_size = self.grid.size()
        # For backwards compatibility with MATLAB, it's necessary to transpose the matrix
        random_noise = np.transpose(np.random.standard_normal((grid_size[1], grid_size[0])))

        return self.sigma[:, :, fixation[0], fixation[1]] * random_noise + self.mu[:, :, fixation[0], fixation[1]]
