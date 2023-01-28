from . import constants
from .ivsn_model import image_preprocessing, IVSN, compute_scanpaths, utils
from os import path

""" Runs the IVSN model on a given dataset.
    Running order is image_preprocessing.py first, IVSN.py next, and compute_scanpaths.py last.
"""

def main(dataset_name, human_subject=None):
    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
    output_path  = path.join(constants.RESULTS_PATH, dataset_name + '_dataset', 'IVSN')

    # Load dataset information
    dataset_info      = utils.load_from_dataset(dataset_path, 'dataset_info.json')
    trials_properties = utils.load_from_dataset(dataset_path, 'trials_properties.json')

    images_dir        = path.join(dataset_path, dataset_info['images_dir'])
    targets_dir       = path.join(dataset_path, dataset_info['targets_dir'])
    max_fixations     = dataset_info['max_scanpath_length']
    images_size       = (dataset_info['image_height'], dataset_info['image_width'])
    receptive_size    = [min(dataset_info['mean_target_size']), min(dataset_info['mean_target_size'])]
    dataset_full_name = dataset_info['dataset_name']

    # For computing different metrics; used only through argument --h
    human_scanpaths_dir = path.join(dataset_path, dataset_info['scanpaths_dir'])
    human_scanpaths     = utils.load_human_scanpaths(human_scanpaths_dir, human_subject)
    if human_scanpaths:
        human_subject_str = '0' + str(human_subject) if human_subject < 10 else str(human_subject)

        #receptive_size    = dataset_info['receptive_size']
        output_path       = path.join(output_path, 'subjects_predictions', 'subject_' + human_subject_str)
        trials_properties = utils.keep_human_trials(human_scanpaths, trials_properties)

    preprocessed_images_dir = path.join(constants.PREPROCESSED_IMAGES_PATH, dataset_full_name)

    print('Preprocessing images...')
    image_preprocessing.chop_images(images_dir, preprocessed_images_dir, images_size, trials_properties)
    print('Running model...')
    IVSN.run(trials_properties, targets_dir, preprocessed_images_dir)
    print('Computing scanpaths...')
    compute_scanpaths.parse_model_data(preprocessed_images_dir, trials_properties, human_scanpaths, images_size, max_fixations, receptive_size, dataset_full_name, output_path)