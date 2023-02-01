from os import path
DATASETS_PATH = 'Datasets'
MODELS_PATH   = path.join('Models', 'IBS', 'configs')
METRICS_PATH  = 'Metrics'
RESULTS_PATH  = 'Results'

AVAILABLE_METRICS = ['perf', 'mm', 'hsp'] # Cumulative performance; Multi-Match; Human scanpath prediction

