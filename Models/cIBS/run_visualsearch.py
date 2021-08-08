import argparse
import visualsearch
import sys
from scripts import loader, constants

" Runs visualsearch/main.py according to the supplied parameters "

def main(config_name, image_name, image_range, number_of_processes, save_probability_maps):
    dataset_info      = loader.load_dataset_info(constants.DATASET_INFO_FILE)
    output_path       = loader.create_output_folders(dataset_info['save_path'], config_name, image_name, image_range)
    checkpoint        = loader.load_checkpoint(output_path)
    config            = loader.load_config(constants.CONFIG_DIR, config_name, constants.IMAGE_SIZE, number_of_processes, save_probability_maps, checkpoint)
    trials_properties = loader.load_trials_properties(dataset_info['trials_properties_file'], image_name, image_range, checkpoint)

    visualsearch.run(config, dataset_info, trials_properties, output_path, constants.SIGMA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the Visual Search model')
    parser.add_argument('--cfg', '--config', type=str, default='default', help='Name of configuration setup. Examples: greedy, ssim, ivsn. Default is bayesian, with correlation \
        and deepgaze as prior.', metavar='cfg')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--img', '--image_name', type=str, default=None, help='Name of the image on which to run the model', metavar='img')
    group.add_argument('--rng', '--range', type=int, nargs=2, default=None, help='Range of image numbers on which to run the model. \
         For example, 1 100 runs the model on the image 1 through 100', metavar='rng')
    parser.add_argument('--m', '--multiprocess', nargs='?', const='all', default=1, \
         help='Number of processes on which to run the model. Leave blank to use all cores available.')
    parser.add_argument('--s', '--save_prob_map', action='store_true', \
         help='Save probability map for each saccade')

    args = parser.parse_args()

    if (isinstance(args.m, str) and args.m != 'all') and int(args.m) < 1:
        print('Invalid value for --multiprocess argument')
        sys.exit(-1)

    main(args.cfg, args.img, args.rng, args.m, args.s)