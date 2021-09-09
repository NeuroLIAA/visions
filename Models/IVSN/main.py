import IVSN
import image_preprocessing
import compute_scanpaths
import argparse
import constants
import json
from os import path

dataset_name = 'IVSN Natural Design Dataset'
images_dir = '../../Datasets/IVSN/stimuli/'
target_dir = '../../Datasets/IVSN/target/'
trials_properties_file = '../../Datasets/IVSN/trials_properties.json'
save_path = '../../Results/IVSN_dataset/IVSN/'
images_size = (1024, 1280)
max_fixations  = 80
receptive_size = 200

# dataset_name = 'cIBS Dataset'
# images_dir = '../../Datasets/cIBS/images/'
# target_dir = '../../Datasets/cIBS/templates/'
# trials_properties_file = '../../Datasets/cIBS/trials_properties.json'
# save_path = '../../Results/cIBS_dataset/IVSN/'
# images_size = (768, 1024)
# max_fixations  = 16
# receptive_size = 72

# dataset_name = 'COCOSearch18 Dataset'
# images_dir = '../../Datasets/COCOSearch18/images/'
# target_dir = '../../Datasets/COCOSearch18/templates/'
# trials_properties_file = '../../Datasets/COCOSearch18/trials_properties.json'
# save_path = '../../Results/COCOSearch18_dataset/IVSN/'
# images_size = (1050, 1680)
# max_fixations  = 10
# receptive_size = 54

def main(dataset_name):
    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
    output_path  = path.join(constants.RESULTS_PATH, path.join(dataset_name + '_dataset', 'IVSN'))

    # Load dataset information
    dataset_info      = utils.load_from_dataset(dataset_path, 'dataset_info.json')
    trials_properties = utils.load_from_dataset(dataset_path, 'trials_properties.json')    

    images_dir     = path.join(dataset_path, dataset_info['images_dir'])
    targets_dir    = path.join(dataset_path, dataset_info['targets_dir'])
    max_fixations  = path.join(dataset_path, dataset_info['max_scanpath_length'])
    images_size    = (dataset_info['image_height'], dataset_info['image_width'])
    receptive_size = dataset_info['receptive_size']

    preprocessed_images_dir = path.join('chopped_images', dataset_name + ' Dataset')

    print('Preprocessing images...')
    image_preprocessing.chop_images(images_dir, preprocessed_images_dir, images_size, trials_properties)
    print('Running model...')
    IVSN.run(images_dir, target_dir, preprocessed_images_dir, trials_properties)
    print('Computing scanpaths...')
    compute_scanpaths.parse_model_data(images_dir, preprocessed_images_dir, images_size, max_fixations, receptive_size, output_path, trials_properties, dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the IVSN visual search model')
    parser.add_argument('-dataset', type=str, help='Name of the dataset on which to run the model. Value must be one of cIBS, COCOSearch18, IVSN or MCS.')

    args = parser.parse_args()
    main(args.dataset)