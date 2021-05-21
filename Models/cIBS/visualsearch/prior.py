import numpy as np
from .utils import utils

def load(image_name, image_size, prior_name, prior_dir):
    " Returns initial probability of the target being there for each position in the image "
    """ Input:
            image_name (string)   : name of the image
            image_size (int, int) : size of the image
            prior_name (string)   : what to use as prior (possible values are deepgaze, center, icf, etc.)
            prior_dir  (string)   : where to look for the prior images. It uses the prior_name as subdirectory
            cell_size  (int, int) : size of the cells in the grid
        Output:
            prior (2D array) : prior corresponding to the image, of the same size
    """
    prior_path = prior_dir + prior_name + '/'
    if prior_name == 'noisy':
        prior = utils.add_white_gaussian_noise(np.ones(shape=image_size), snr_db=25)
    else:
        # TODO: Agregar código para generar el prior, de hacer falta
        prior = utils.load_image(prior_path, image_name)

    # Normalize values
    prior = prior / np.max(prior)

    return prior

# TODO: Definir para qué sirve la función y asignarle mejores nombres
def sum(prior, max_saccades):
    """ Input:
            prior (2D array)   : prior where probabilites will be summed
            max_saccades (int) : maximum possible number of saccades
        Output:
            ????
    """
    prior_size = (prior.shape[0], prior.shape[1])
    number_of_probs  = prior_size[0] * prior_size[1]
    sum_of_all_probs = number_of_probs * max_saccades

    prior_probs = prior * (sum_of_all_probs - number_of_probs) / np.sum(prior) + 1
    
    return prior_probs