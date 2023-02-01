from os import path
# Configuration constants
# All paths are relative to root
CONFIG_DIR    = path.join('Models', 'IBS', 'configs')
DATASETS_PATH = 'Datasets'
RESULTS_PATH  = 'Results'
SALIENCY_PATH = path.join('Models', 'IBS', 'data', 'saliency')
TARGET_SIMILARITY_PATH = path.join('Models', 'IBS', 'data', 'target_similarity_maps')

SIGMA      = [[4000, 0], [0, 2600]]
IMAGE_SIZE = (768, 1024)