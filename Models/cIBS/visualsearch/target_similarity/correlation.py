import numpy as np
from skimage.feature import match_template

""" Computes target similarity for each possible fixation in the grid by using normalized cross correlation
"""

class Correlation():
    def __init__(self, image, target, visibility_map, scale_factor, additive_shift, grid, target_bbox, seed):
        # Set the seed for generating random noise
        np.random.seed(seed)

        self.grid_size = grid.size()
        self.sigma, self.mu = self.create_target_similarity_map(image, target, visibility_map, scale_factor, additive_shift, grid, target_bbox)
    
    def create_target_similarity_map(self, image, target, visibility_map, scale_factor, additive_shift, grid, target_bbox):
        " Creates a target similarity map for a given visibility map and target, using cross correlation. "
        " The output is a normal distribution, which depends upon target similarity and visibility "
        """ Input:
                image  (2D array)    : image where the target is
                target (2D array)    : target image
                visibility_map (VisibilityMap) : visibility map which indicates how focus decays over distance from the fovea
                scale_factor   (int) : modulates the inverse of the visibility and prevents the variance from diverging
                additive_shift (int) : modulates the inverse of the visibility and prevents the variance from diverging
                grid (Grid) : representation of an image with cells instead of pixels 
                target_bbox (int array of length four) : target bounding box (top left row, top left column, bottom right row, bottom right column) represented as cells in the grid 
            Output:
                sigma, mu (4D arrays) : values of the normal distribution for each possible fixation in the grid
        """
        grid_size = grid.size()
        # Initialize mu, where each cell has a value of 0.5 if the target is present and -0.5 otherwise
        mu = np.ones(shape=(grid_size[0], grid_size[1], grid_size[0], grid_size[1])) * (-0.5)
        for row in range(target_bbox[0], target_bbox[2] + 1):
            for column in range(target_bbox[1], target_bbox[3] + 1):
                mu[row, column] = np.ones(shape=grid_size) * 0.5
        
        cross_correlation = match_template(image, target, pad_input=True)
        if len(cross_correlation.shape) > 2:
            # If it's coloured, convert to single channel using skimage rgb2gray formula
            cross_correlation = cross_correlation[:, :, 0] * 0.2125 + \
                                cross_correlation[:, :, 1] * 0.7154 + \
                                cross_correlation[:, :, 2] * 0.0721
        # Reduce to grid
        cross_correlation = grid.reduce(cross_correlation, mode='max')

        # Convert values to the interval [-0.5, 0.5] 
        cross_correlation = cross_correlation - np.min(cross_correlation)
        cross_correlation = cross_correlation / np.max(cross_correlation) - 0.5
        # Make it the same shape as mu
        cross_correlation = np.tile(cross_correlation[:, :, np.newaxis, np.newaxis], (1, 1, grid_size[0], grid_size[1]))

        # Modify mu in order to incorporate target similarity and visibility
        mu = mu * (visibility_map.normalized_at_every_fixation() + 0.5) + cross_correlation * (1 - visibility_map.normalized_at_every_fixation() + 0.5)
        # Convert values to the interval [-1, 1]
        mu = mu / 2

        sigma = np.ones(shape=mu.shape)
        # Variance now depends on the visibility
        sigma = sigma / (visibility_map.normalized_at_every_fixation() * scale_factor + additive_shift)

        return sigma, mu

    def at_fixation(self, fixation):
        " Given a fixation in the grid, it returns the target similarity map, represented as a 2D array of scalars, with added random noise "
        """ Input:
                fixation (int, int) : cell in the grid on which the observer is fixating
            Output:
                target_similarity_map (2D array of floats) : matrix the size of the grid, where each value is a scalar which represents how similar the position is to the target
        """
        # For backwards compatibility with MATLAB, it's necessary to transpose the matrix
        random_noise = np.transpose(np.random.standard_normal((self.grid_size[1], self.grid_size[0])))

        return self.sigma[:, :, fixation[0], fixation[1]] * random_noise + self.mu[:, :, fixation[0], fixation[1]]