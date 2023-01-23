from pathlib import Path

DATASETS_PATH = 'Datasets'
RESULTS_PATH  = 'Results'
CFG_FILE      = Path('Models', 'eccNET', 'visualsearch', 'config.json')
VGG16_WEIGHTS = Path('Models', 'eccNET', 'pretrained_model', 'vgg16_imagenet_filters.h5')

EYE_RES = 736
IMG_HEIGHT, IMG_WIDTH = 736, 896
TARGET_HEIGHT, TARGET_WIDTH = 32, 32
RECEPTIVE_FIELD = 2*TARGET_HEIGHT
DEG2PIXEL = 30
CORNER_BIAS = 16*4
DOG_SIZE  = [[3, 0], [5, 1]]