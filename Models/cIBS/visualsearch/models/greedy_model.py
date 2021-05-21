import numpy as np

class GreedyModel:
    def __init__(self):
        pass

    def next_fixation(self, posterior):
        " Given the posterior for each cell in the grid, this function computes the next fixation by searching for the maximum values from it "
        """ Input:
                posterior (2D array of floats) : matrix the size of the grid containing the posterior probability for each cell
            Output:
                next_fix (int, int) : cell chosen to be the next fixation
        """
        coordinates = np.where(posterior == np.amax(posterior))
        next_fix    = (coordinates[0][0], coordinates[1][0])

        return next_fix