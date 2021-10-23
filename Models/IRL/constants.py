from os import path
# All paths are relative to root
DATASETS_PATH = 'Datasets'
RESULTS_PATH  = 'Results'
DCBS_PATH     = path.join(path.join('Models', 'IRL'), 'dataset_root')
HPARAMS_PATH  = path.join(path.join('Models', 'IRL'), 'hparams')
TRAINED_MODELS_PATH = path.join(path.join('Models', 'IRL'), 'trained_models')

NUMBER_OF_BELIEF_MAPS = 134
# Sigma of the gaussian filter applied to the search image
SIGMA_BLUR = 2