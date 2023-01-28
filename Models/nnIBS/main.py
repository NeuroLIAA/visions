from .scripts import loader, constants
from . import visualsearch
from os import path

" Runs visualsearch/main.py according to the supplied parameters "

def setup_and_run(dataset_name, config_name, image_name, image_range, human_subject, number_of_processes, save_probability_maps):
    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
    output_path  = path.join(constants.RESULTS_PATH, path.join(dataset_name + '_dataset', 'nnIBS'))

    trials_properties_file = path.join(dataset_path, 'trials_properties.json')

    dataset_info      = loader.load_dataset_info(dataset_path)
    output_path       = loader.create_output_folders(output_path, config_name, image_name, image_range, human_subject)
    checkpoint        = loader.load_checkpoint(output_path)
    human_scanpaths   = loader.load_human_scanpaths(dataset_info['scanpaths_dir'], human_subject)
    config            = loader.load_config(constants.CONFIG_DIR, config_name, constants.IMAGE_SIZE, dataset_info['max_scanpath_length'], number_of_processes, save_probability_maps, human_scanpaths, checkpoint)
    trials_properties = loader.load_trials_properties(trials_properties_file, image_name, image_range, human_scanpaths, checkpoint)

    visualsearch.run(config, dataset_info, trials_properties, human_scanpaths, output_path, constants.SIGMA)

""" Main method, added to be polymorphic with respect to the other models """
def main(dataset_name, human_subject=None):
    setup_and_run(dataset_name, config_name=constants.CONFIG_NAME, image_name=None, image_range=None, human_subject=human_subject, number_of_processes=constants.NUMBER_OF_PROCESSES, save_probability_maps=False)