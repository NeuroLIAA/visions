import numpy as np

""" TODO: agregar descripción sobre el comportamiento de este en particular, o de dónde sale (Geisler et al 2005?)
"""

class Geisler():
    def __init__(self, visibility_map, scale_factor, additive_shift, grid_size, target_bbox, seed):
        # Set the seed for generating random noise
        np.random.seed(seed)
        
        self.grid_size = grid_size
        self.sigma, self.mu = self.create_target_similarity_map(visibility_map, scale_factor, additive_shift, grid_size, target_bbox)

    def create_target_similarity_map(self, visibility_map, scale_factor, additive_shift, grid_size, target_bbox):
        " Creates a target similarity map for a given visibility map and target. The output's shape is grid_size x grid_size "
        """ Input:
                visibility_map (VisibilityMap) : visibility map which indicates how focus decays over distance from the fovea
                scale_factor   (int) : modulates the inverse of the visibility and prevents the variance from diverging
                additive_shift (int) : modulates the inverse of the visibility and prevents the variance from diverging
                grid_size (int, int) : size of the grid
                target_bbox (int array of length four) : target bounding box (top left row, top left column, bottom right row, bottom right column) represented as cells in the grid 
            Output:
                sigma, mu (4D arrays) : values of the normal distribution for each possible fixation in the grid
        """
        # Initialize mu, where each cell has a value of 0.5 if the target is present and -0.5 otherwise
        mu = np.ones(shape=(grid_size[0], grid_size[1], grid_size[0], grid_size[1])) * (-0.5)
        for row in range(target_bbox[0], target_bbox[2] + 1):
            for column in range(target_bbox[1], target_bbox[3] + 1):
                mu[row, column] = np.ones(shape=grid_size) * 0.5
        
        # Initialize sigma
        sigma = np.ones(shape=mu.shape)
        # Variance now depends on the visibility
        sigma = sigma / (visibility_map.normalized_at_every_fixation() * scale_factor + additive_shift)

        return sigma, mu

    def at_fixation(self, fixation):
        " Given a fixation in the grid, it returns the target similarity map, represented as a 2D array of scalars with added random noise "
        """ Input:
                fixation (int, int) : cell in the grid on which the observer is fixating
            Output:
                target_similarity_map (2D array of floats) : matrix the size of the grid, where each value is a scalar which represents how similar the position is to the target
        """
        # For backwards compatibility with MATLAB, it's necessary to transpose the matrix
        random_noise = np.transpose(np.random.standard_normal((self.grid_size[1], self.grid_size[0])))

        #return self.sigma[:, :, fixation[0], fixation[1]] * self.random_generator.standard_normal(self.grid_size) + self.mu[:, :, fixation[0], fixation[1]]
        return self.sigma[:, :, fixation[0], fixation[1]] * random_noise + self.mu[:, :, fixation[0], fixation[1]]
        
