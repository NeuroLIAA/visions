from ivsn_model import image_preprocessing, IVSN, compute_scanpaths
import argparse
import constants
import json
from os import path

""" Runs the IVSN model on a given dataset.
    Running order is image_preprocessing.py first, IVSN.py next, and compute_scanpaths.py last.
"""

def main(dataset_name):
    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
    output_path  = path.join(constants.RESULTS_PATH, path.join(dataset_name + '_dataset', 'IVSN'))

    # Load dataset information
    dataset_info      = utils.load_from_dataset(dataset_path, 'dataset_info.json')
    trials_properties = utils.load_from_dataset(dataset_path, 'trials_properties.json')    

    images_dir        = path.join(dataset_path, dataset_info['images_dir'])
    targets_dir       = path.join(dataset_path, dataset_info['targets_dir'])
    max_fixations     = path.join(dataset_path, dataset_info['max_scanpath_length'])
    images_size       = (dataset_info['image_height'], dataset_info['image_width'])
    receptive_size    = dataset_info['receptive_size']
    dataset_full_name = dataset_info['dataset_name']

    preprocessed_images_dir = path.join('chopped_images', dataset_full_name)

    print('Preprocessing images...')
    image_preprocessing.chop_images(images_dir, preprocessed_images_dir, images_size, trials_properties)
    print('Running model...')
    IVSN.run(images_dir, target_dir, preprocessed_images_dir, trials_properties)
    print('Computing scanpaths...')
    compute_scanpaths.parse_model_data(images_dir, preprocessed_images_dir, images_size, max_fixations, receptive_size, output_path, trials_properties, dataset_full_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the IVSN visual search model')
    parser.add_argument('-dataset', type=str, help='Name of the dataset on which to run the model. Value must be one of cIBS, COCOSearch18, IVSN or MCS.')

    args = parser.parse_args()
    main(args.dataset)