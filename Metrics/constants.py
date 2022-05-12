from os import path
# All paths are relative to root
RESULTS_PATH  = 'Results'
DATASETS_PATH = 'Datasets'
MODELS_PATH   = 'Models'

RANDOM_SEED = 1234
MAX_DIR_SIZE = 10 # MBytes, for probability maps dirs

FILENAME = 'Metrics.json'

# Constants for center bias model
CENTER_BIAS_PATH      = path.join('Metrics', 'center_bias')
CENTER_BIAS_FIXATIONS = path.join(CENTER_BIAS_PATH, 'cat2000_fixations.json')
CENTER_BIAS_SIZE      = (1080, 1920)

# To ensure models have the same color in the plots across all datasets
MODELS_COLORS = ['#2ca02c', '#ff7f0e', '#d62728']
HUMANS_COLOR  = '#1f77b4'
