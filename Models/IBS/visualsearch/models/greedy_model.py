import numpy as np
from ..utils import utils

class GreedyModel:
    def __init__(self, save_probability_maps):
        self.save_probability_maps = save_probability_maps

    def next_fixation(self, posterior, image_name, fixation_number, output_path):
        " Given the posterior for each cell in the grid, this function computes the next fixation by searching for the maximum values from it "
        """ Input:
                posterior (2D array of floats) : matrix the size of the grid containing the posterior probability for each cell
            Output:
                next_fix (int, int) : cell chosen to be the next fixation
            
            (The rest of the input arguments are used to save the probability map to a CSV file.)
        """
        coordinates = np.where(posterior == np.amax(posterior))
        next_fix    = (coordinates[0][0], coordinates[1][0])

        if self.save_probability_maps:
            utils.save_probability_map(output_path, image_name, posterior, fixation_number)

        return next_fix