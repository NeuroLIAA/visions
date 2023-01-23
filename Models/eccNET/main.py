from . import constants
from .visualsearch import run_exp, utils
from pathlib import Path

""" Runs the eccNET model on a given dataset """

def main(dataset_name, human_subject=None):
    if human_subject is not None:
        raise NotImplementedError("Human scanpath prediction is not implemented yet")
    
    dataset_path = Path(constants.DATASETS_PATH) / dataset_name
    output_path  = Path(constants.RESULTS_PATH) / (dataset_name + '_dataset') / 'eccNET'

    # Load dataset information
    dataset_info      = utils.load_from_dataset(dataset_path, 'dataset_info.json')
    trials_properties = utils.load_from_dataset(dataset_path, 'trials_properties.json')

    dataset_fullname = dataset_info['dataset_name']
    images_dir    = dataset_path / dataset_info['images_dir']
    targets_dir   = dataset_path / dataset_info['targets_dir']
    max_fixations = dataset_info['max_scanpath_length']
    img_size      = (constants.IMG_HEIGHT, constants.IMG_WIDTH)
    target_size   = (constants.TARGET_HEIGHT, constants.TARGET_WIDTH)
    num_images    = len(trials_properties)

    # Create data structure for visual search model
    exp_info = utils.build_expinfo(num_images, max_fixations, img_size, target_size, constants.EYE_RES, constants.DEG2PIXEL, constants.DOG_SIZE)
    exp_info['corner_bias'] = constants.CORNER_BIAS

    run_exp.start(trials_properties, exp_info, images_dir, targets_dir, constants.CFG_FILE, constants.VGG16_WEIGHTS, dataset_fullname, output_path)