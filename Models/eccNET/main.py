from . import constants
from .visualsearch import run_exp, utils
from pathlib import Path

""" Runs the eccNET model on a given dataset """

def main(dataset_name, human_subject=None):
    dataset_path = Path(constants.DATASETS_PATH) / dataset_name
    output_path  = Path(constants.RESULTS_PATH) / (dataset_name + '_dataset') / 'eccNET'

    # Load dataset information
    dataset_info      = utils.load_from_dataset(dataset_path, 'dataset_info.json')
    trials_properties = utils.load_from_dataset(dataset_path, 'trials_properties.json')

    dataset_fullname = dataset_info['dataset_name']
    images_dir     = dataset_path / dataset_info['images_dir']
    targets_dir    = dataset_path / dataset_info['targets_dir']
    max_fixations  = dataset_info['max_scanpath_length']
    img_size       = (dataset_info['image_height'], dataset_info['image_width'])
    model_img_size = (constants.IMG_HEIGHT, constants.IMG_WIDTH)
    target_size    = (constants.TARGET_HEIGHT, constants.TARGET_WIDTH)
    num_images     = len(trials_properties)

    # Create data structure for visual search model
    exp_info = utils.build_expinfo(num_images, max_fixations, model_img_size, target_size, constants.EYE_RES, constants.DEG2PIXEL, constants.DOG_SIZE)
    exp_info['corner_bias'] = constants.CORNER_BIAS

    # For computing hsp (human scanpath prediction) metrics
    human_scanpaths_dir = dataset_path / dataset_info['scanpaths_dir']
    human_scanpaths     = utils.load_human_scanpaths(human_subject, img_size, model_img_size, human_scanpaths_dir)
    if human_scanpaths:
        human_subject_str = str(human_subject).zfill(2)
        output_path       = output_path / 'subjects_predictions' / ('subject_' + human_subject_str)
        trials_properties = utils.keep_human_trials(human_scanpaths, trials_properties)

    run_exp.start(trials_properties, exp_info, images_dir, targets_dir, human_scanpaths, constants.CFG_FILE, constants.VGG16_WEIGHTS, dataset_fullname, output_path)