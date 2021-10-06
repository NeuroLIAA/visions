import constants
import argparse
from scripts.multimatch import Multimatch
from scripts.human_scanpath_prediction import HumanScanpathPrediction
from scripts.cumulative_performance import Cumulative_performance
from scripts import utils
from os import listdir, path

def main(compute_cumulative_performance, compute_multimatch, compute_human_scanpath_prediction):
    dataset_results_dirs = listdir(constants.RESULTS_DIR)
    for dataset in dataset_results_dirs:
        dataset_name = dataset.split('_')[0]
        dataset_path = path.join(constants.DATASETS_DIR, dataset_name)
        dataset_info = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))

        human_scanpaths_dir = path.join(dataset_path, dataset_info['scanpaths_dir'])
        dataset_results_dir = path.join(constants.RESULTS_DIR, dataset)

        max_scanpath_length = dataset_info['max_scanpath_length']
        number_of_images    = dataset_info['number_of_images']

        # Define subset of images for COCOSearch18
        if dataset_name == 'COCOSearch18':
            number_of_images = constants.COCOSEARCH_SUBSET_SIZE

        # Initialize objects
        multimatch = Multimatch(dataset_name, human_scanpaths_dir, dataset_results_dir, number_of_images, compute_multimatch)

        subjects_cumulative_performance = Cumulative_performance(dataset_name, number_of_images, max_scanpath_length, compute_cumulative_performance)
        subjects_cumulative_performance.add_human_mean(human_scanpaths_dir, constants.HUMANS_COLOR)

        human_scanpath_prediction = HumanScanpathPrediction(dataset_name, human_scanpaths_dir, dataset_results_dir, constants.MODELS_DIR, compute_human_scanpath_prediction)

        # Compute models metrics and compare them with human subjects metrics
        models = listdir(dataset_results_dir)
        color_index = 0
        for model_name in models:
            if not(path.isdir(path.join(dataset_results_dir, model_name))):
                continue

            model_scanpaths_file = path.join(path.join(dataset_results_dir, model_name), 'Scanpaths.json')
            model_scanpaths      = utils.load_dict_from_json(model_scanpaths_file)

            subjects_cumulative_performance.add_model(model_name, model_scanpaths, constants.MODELS_COLORS[color_index])

            # Human multimatch scores are different for each model, since each model uses different image sizes
            multimatch.load_human_mean_per_image(model_name, model_scanpaths)
            multimatch.add_model_vs_humans_mean_per_image(model_name, model_scanpaths, constants.MODELS_COLORS[color_index])

            human_scanpath_prediction.compute_metrics_for_model(model_name)

            color_index += 1

        subjects_cumulative_performance.save_results(save_path=dataset_results_dir, filename=constants.FILENAME)
        multimatch.save_results(save_path=dataset_results_dir, filename=constants.FILENAME)

        subjects_cumulative_performance.plot(save_path=dataset_results_dir)
        multimatch.plot(save_path=dataset_results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute all metrics for each visual search model on each dataset. To only run a subset of them, specify by argument which ones.')
    parser.add_argument('--perf', '--performance', action='store_true', help='Compute cumulative performance')
    parser.add_argument('--mm', '--multimatch', action='store_true', help='Compute multimatch on the models')
    parser.add_argument('--hsp', '--human_scanpath_prediction', action='store_true', help='Compute human scanpath prediction on the models. \
        See "Kümmerer, M. & Bethge, M. (2021), State-of-the-Art in Human Scanpath Prediction" for more information. WARNING: If not precomputed, EXTREMELY SLOW!')

    args = parser.parse_args()

    # If none were specified, default to True
    if not (args.perf or args.mm or args.hsp):
        args.perf = True
        args.mm   = True
        args.hsp  = True

    main(args.perf, args.mm, args.hsp)