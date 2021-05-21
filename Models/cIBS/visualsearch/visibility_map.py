import numpy as np
from scipy.stats import multivariate_normal

" The visibility map represents how focus decays over distance from the fovea "
" This implementation uses the gaussian distribution, where the mean values correspond to the center of the fixation in pixels "
" The covariance matrix was calculated before hand by estimating the vision angle of the fovea to the screen in the human experiments "

class VisibilityMap:
    def __init__(self, image_size, grid, sigma):
        self.visibility_map = self.create(image_size, grid, sigma)
        self.visibility_map_normalized = self.convert_to_unit_interval()

    def create(self, image_size, grid, sigma):
        " Creates a visibility map for a given image and cell size. The output's shape is grid size x grid size."
        """ Input:
                image_size (int, int) : height and width of the image, respectively, in pixels
                grid  (Grid)          : representation of the image in cells
                sigma (2D array)      : covariance matrix of the multivariate normal
            Output:
                visibility_map [a, b, c, d] (4D array) : where visibility_map[:, :, c, d] is the visibility map of fixing the view in position (c, d)
        """
        visibility_map = np.empty(shape=grid.size() + grid.size())

        x_range = np.linspace(0, image_size[0], grid.size()[0])
        y_range = np.linspace(0, image_size[1], grid.size()[1])

        x_matrix, y_matrix = np.meshgrid(x_range, y_range)
        for row in range(grid.size()[0]):
            for column in range(grid.size()[1]):
                fixation = grid.map_cell_to_pixels((row, column))

                quantiles = np.transpose([y_matrix.flatten(), x_matrix.flatten()])
                mvn_at_fixation = multivariate_normal.pdf(quantiles, mean=[fixation[1], fixation[0]], cov=sigma)
                mvn_at_fixation = np.reshape(mvn_at_fixation, grid.size(), order='F')

                visibility_map[row, column] = mvn_at_fixation
        
        # TODO: Explicar para qué sirven estas cuentas
        visibility_map = visibility_map - np.min(visibility_map)
        visibility_map = visibility_map / np.max(visibility_map) * 3

        return visibility_map
    
    def convert_to_unit_interval(self):
        " Returns a matrix where values go from zero to one for each fixation, where one corresponds to the maximum value in visibility_map at the corresponding fixation "
        grid_size = (self.visibility_map.shape[0], self.visibility_map.shape[1])
        visibility_map_normalized = np.empty(shape=(grid_size[0], grid_size[1], grid_size[0], grid_size[1]))
        for row in range(grid_size[0]):
            for column in range(grid_size[1]):
                visibility_map_normalized[:, :, row, column] = self.visibility_map[:, :, row, column] - np.min(self.visibility_map[:, :, row, column])
                visibility_map_normalized[:, :, row, column] = self.visibility_map[:, :, row, column] / np.max(self.visibility_map[:, :, row, column])
        
        return visibility_map_normalized

    def at_fixation(self, fixation):
        " Given a fixation in the grid, it returns the visibility map, represented as a 2D array of scalars "
        """ Input:
                fixation (int, int) : cell in the grid on which the observer is fixating
            Output:
                visibility_map (2D array of floats) : matrix the size of the grid, where each value is a scalar which represents how much the view diminishes
        """
        return self.visibility_map[:, :, fixation[0], fixation[1]]
    
    def normalized_at_every_fixation(self):
        " It returns the visibility map for every fixation possible, represented as a 4D array, where each value goes from zero to one"
        return self.visibility_map_normalized