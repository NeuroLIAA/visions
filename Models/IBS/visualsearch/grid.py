import numpy as np

class Grid:
    def __init__(self, image_size, cell_size):
        self.cell_size = cell_size
        self.grid_size = -(image_size // -cell_size)
        self.offset    = image_size % cell_size

    def size(self):
        return tuple(self.grid_size)

    def reduce(self, image, mode):
        " Given an image, its dimensions are reduced by a factor of cell_size. The new values in each position of the grid correspond to the mean or max values of that portion of the image "
        """ Input:
                image (2D array) : image to be reduced to a grid
                mode (string)    : how to reduce values to a cell, it can be either 'mean' or 'max'
            Output:
                grid (2D array)  : matrix where each position contains the mean or max value of the corresponding portion of the image of size cell_size
        """
        grid = np.empty(shape=self.grid_size)
        
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                start_row, start_column = np.array([row, column]) * self.cell_size 
                end_row, end_column     = np.array([start_row, start_column]) + self.cell_size
                
                # Check for edge cases
                if self.offset[0] and row == self.grid_size[0] - 1:
                    end_row    = start_row + self.offset[0]
                if self.offset[1] and column == self.grid_size[1] - 1:
                    end_column = start_column + self.offset[1]

                if mode == 'mean':
                    grid[row, column] = np.mean(image[start_row:end_row, start_column:end_column])
                elif mode == 'max':
                    grid[row, column] = np.max(image[start_row:end_row, start_column:end_column])
        
        return grid

    def map_to_cell(self, pixel):
        " Given a pixel in the image, this function returns the corresponding coordinate of the pixel in the grid "
        """ Input:
                pixel (int, int) : row and column number of the pixel in the image
            Output:
                cell (int, int)  : corresponding cell of the pixel in the image
        """
        return np.array(pixel, dtype=int) // self.cell_size

    def map_cell_to_pixels(self, cell):
        " Given a cell in the grid, this function returns the corresponding pixel in the image, centered in the cell "
        """ Input:
                cell (int, int)  : row and column of the cell in the gird
            Output:
                pixel (int, int) : corresponding pixel at the center of the cell in the image
        """
        cell_center = np.array(cell) * self.cell_size
        
        # If the cell is outside of the image, use the offset
        if self.offset[0] and cell[0] == self.grid_size[0] - 1:
            cell_center[0] = cell_center[0] + self.offset[0] - 1
        else:
            cell_center[0] = cell_center[0] + self.cell_size // 2

        if self.offset[1] and cell[1] == self.grid_size[1] - 1:
            cell_center[1] = cell_center[1] + self.offset[1] - 1
        else:
            cell_center[1] = cell_center[1] + self.cell_size // 2

        return cell_center