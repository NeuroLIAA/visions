from .scripts import loader, constants
import argparse
from . import visualsearch
import sys
from os import path

" Runs visualsearch/main.py according to the supplied parameters "

def setup_and_run(dataset_name, config_name, image_name, image_range, human_subject, number_of_processes, save_probability_maps):
    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
    output_path  = path.join(constants.RESULTS_PATH, path.join(dataset_name + '_dataset', 'sIBS'))

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
    setup_and_run(dataset_name, config_name='default', image_name=None, image_range=None, human_subject=human_subject, number_of_processes='all', save_probability_maps=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the sIBS Visual Search model')
    parser.add_argument('-dataset', type=str, help='Name of the dataset on which to run the model. Value must be one of Interiors, COCOSearch18, Unrestricted or MCS.')
    parser.add_argument('--cfg', '--config', type=str, default='default', help='Name of configuration setup. Examples: greedy, ssim, ivsn. Default is bayesian, with correlation \
        and deepgaze as prior.', metavar='cfg')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--img', '--image_name', type=str, default=None, help='Name of the image on which to run the model', metavar='img')
    group.add_argument('--rng', '--range', type=int, nargs=2, default=None, help='Range of image numbers on which to run the model. \
         For example, 1 100 runs the model on the image 1 through 100', metavar='rng')
    parser.add_argument('--m', '--multiprocess', nargs='?', const='all', default=1, \
         help='Number of processes on which to run the model. Leave blank to use all cores available.')
    parser.add_argument('--s', '--save_prob_map', action='store_true', \
         help='Save probability map for each saccade. If human_subject is provided, this will always be true.')
    parser.add_argument('--h', '--human_subject', type=int, default=None, help='Human subject on which the model will follow its scanpaths, saving the probability map for each saccade.\
         Useful for computing different metrics. See "Kümmerer, M. & Bethge, M. (2021), State-of-the-Art in Human Scanpath Prediction" for more information')

    args = parser.parse_args()

    if (isinstance(args.m, str) and args.m != 'all') and int(args.m) < 1:
        print('Invalid value for --multiprocess argument')
        sys.exit(-1)

    setup_and_run(args.dataset, args.cfg, args.img, args.rng, args.h, args.m, args.s)