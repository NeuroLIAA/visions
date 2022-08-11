import numpy as np
import warnings
from scipy.stats import norm
from scipy.interpolate import interp1d
from multiprocessing import Process, shared_memory
from ..utils import utils
from os import cpu_count

class BayesianModel:
    def __init__(self, grid_size, visibility_map, norm_cdf_tolerance, number_of_processes, save_probability_maps):
        self.grid_size      = grid_size
        self.visibility_map = visibility_map
        self.norm_cdf_table = self.create_norm_cdf_table(norm_cdf_tolerance)
        self.number_of_processes = number_of_processes

        self.save_probability_maps = save_probability_maps
    
    def create_norm_cdf_table(self, norm_cdf_tolerance):
        " Build normal curve table with default mean and variance. Row values go from norm_cdf_tolerance to 1 - norm_cdf_tolerance "
        """ Input:
                norm_cdf_tolerance (float) : value from which to compute the normal distribution probabilities
            Output:
                norm_cdf_table (dict) : table with the cumulative normal distribution values in the 'x' key
        """
        columns = norm.ppf(np.arange(start=norm_cdf_tolerance, stop=1, step=norm_cdf_tolerance))
        rows    = norm.cdf(columns)

        return {'x': columns, 'y': rows}
    
    def next_fixation(self, posterior, image_name, fixation_number, output_path):
        " Computes the next fixation according to the posterior, which size is equal to the grid "
        """ Input:
                posterior (2D array) : probability map of the size of the grid
            Output:
                next_fix (int, int) : cell in the grid which maximizes the probability of being correct about the target being in that cell in regard to the posterior

            (The rest of the input arguments are used to save the probability map to a CSV file.)
        """
        probability_at_each_fixation = np.empty(shape=self.grid_size)

        if self.number_of_processes > 1:
            self.parallelize_probability_computation(probability_at_each_fixation, posterior)
        else:
            self.compute_probability_on_rows(probability_at_each_fixation, posterior, rows=range(self.grid_size[0]))

        if self.save_probability_maps:
            utils.save_probability_map(output_path, image_name, probability_at_each_fixation, fixation_number)
        
        # Get the fixation which maximizes the probability of being correct
        coordinates = np.where(probability_at_each_fixation == np.max(probability_at_each_fixation))
        next_fix    = (coordinates[0][0], coordinates[1][0])

        return next_fix      
    
    def parallelize_probability_computation(self, probability_at_each_fixation, posterior):
        " This method is only executed if self.number_of_processes is greater than one "
        " It divides the computation of the rows of the matrix probability_at_each_fixation in self.number_of_processes processes "
        """ Input: 
                probability_at_each_fixation (2D array) : matrix of the size of the grid which will hold the values of the probability of being correct at each location
                posterior (2D array) : probability map of the size of the grid
        """
        try:
            # Processes will iterate over the rows of probability_at_each_fixation. Divide the rows in equal chunks.
            number_of_procs  = self.number_of_processes
            number_of_rows   = self.grid_size[0]
            remainder        = number_of_rows % number_of_procs
            indexes          = list(range(number_of_rows)) 
            chunks = [indexes[i * (number_of_rows // number_of_procs) + min(i, remainder):(i + 1) * (number_of_rows // number_of_procs) + min(i + 1, remainder)] for i in range(number_of_procs)]

            # Create shared memory and allocate the matrix in it
            shared_mem       = shared_memory.SharedMemory(create=True, size=probability_at_each_fixation.nbytes)
            shared_matrix    = np.ndarray(probability_at_each_fixation.shape, dtype=probability_at_each_fixation.dtype, buffer=shared_mem.buf)
            shared_matrix[:] = probability_at_each_fixation[:]

            procs = []
            for chunk in chunks:
                proc = Process(target=self.proc_compute_probability_on_chunk, args=(shared_mem.name, posterior, chunk, ))
                procs.append(proc)
                proc.start()
            
            # Wait for each process to complete
            for proc in procs:
                proc.join()
            
            # Copy back the data
            probability_at_each_fixation[:] = shared_matrix[:]
        finally:
            # Free and release shared memory
            shared_mem.close()
            shared_mem.unlink()
    
    def proc_compute_probability_on_chunk(self, matrix_pointer, posterior, chunk):
        " This method is executed by each process, were self.number_of_processes to be greater than one "
        " It runs on a subset of the rows on probability_at_each_fixation, which is stored at matrix_pointer "
        try:
            # Attach to existing shared memory block
            shared_mem = shared_memory.SharedMemory(name=matrix_pointer)
            probability_at_each_fixation = np.ndarray(shape=self.grid_size, dtype=np.float64, buffer=shared_mem.buf)

            self.compute_probability_on_rows(probability_at_each_fixation, posterior, chunk)
        except KeyboardInterrupt:
            pass
        finally:
            # Clean up
            shared_mem.close()

    def compute_probability_on_rows(self, probability_at_each_fixation, posterior, rows):
        " Computes the probability of being correct at each fixation on the given subset of rows of the matrix probability_at_each_fixation "
        # Ignore user warnings due to masked values
        warnings.filterwarnings('ignore', category=UserWarning)
        # Ignore invalid and divide by zero warnings, they'll be dealt with later
        np.seterr(divide='ignore', invalid='ignore', over='ignore')

        probability_of_being_correct = np.empty(shape=self.grid_size)
        for possible_nextfix_row in rows: 
            for possible_nextfix_column in range(self.grid_size[1]):
                visibility_map_at_fixation = self.visibility_map.at_fixation((possible_nextfix_row, possible_nextfix_column))
                for possible_target_location_row in range(self.grid_size[0]):
                    for possible_target_location_column in range(self.grid_size[1]):
                        probability_of_being_correct[possible_target_location_row, possible_target_location_column] = \
                            self.compute_conditional_probability(possible_target_location_row, possible_target_location_column, posterior, visibility_map_at_fixation)

                probability_at_each_fixation[possible_nextfix_row, possible_nextfix_column] = np.nansum(posterior * probability_of_being_correct)


    def compute_conditional_probability(self, target_location_row, target_location_column, posterior, visibility_map_at_fixation, alpha=1):
        " Computes the probability of being correct given the visibility map of the next fixation and that the true target location is (target_location_row, target_location_column) "
        posterior_at_target_location  = posterior[target_location_row, target_location_column]
        visibility_at_target_location = visibility_map_at_fixation[target_location_row, target_location_column]

        b = (-2 * np.log(posterior / posterior_at_target_location) + np.square(visibility_map_at_fixation) \
             + np.square(visibility_at_target_location)) / (2 * visibility_map_at_fixation)
        m =  visibility_at_target_location /  visibility_map_at_fixation

        # We ensure the product is only for i != j (normcdf(1000000) = 1)
        m[target_location_row, target_location_column] = 0
        b[target_location_row, target_location_column] = 1000000

        # Check the limits of the integral (normcdf(-20) = 0 and so will be the product)        
        if not visibility_at_target_location:
            min_w = -20
        else:
            min_w = max(np.nanmax((-20 - b[m > 0]) / m[m > 0]), -20)
        
        max_w = 20

        if min_w >= max_w: return 0

        w_range = np.linspace(min_w, max_w, 50)

        values_for_normcdf = np.matmul(m.flatten()[:, np.newaxis], w_range[np.newaxis, :]) + np.tile(b.flatten()[:, np.newaxis], (1, len(w_range)))
        values_for_normcdf[np.isnan(values_for_normcdf)] = 1

        # Use the previously computed normcdf table to get the values needed
        normcdf_at_values = np.interp(values_for_normcdf, self.norm_cdf_table['x'], self.norm_cdf_table['y'])
        #normcdf_at_values = interp1d(self.norm_cdf_table['x'], self.norm_cdf_table['y'], kind='nearest', fill_value='extrapolate')(values_for_normcdf)

        phi_w  = np.exp(-0.5 * np.square(w_range)) / np.sqrt(2 * np.pi)
        points = phi_w[np.newaxis, :] * (np.prod(alpha * normcdf_at_values, axis=0) / alpha)

        return np.trapz(points, w_range)