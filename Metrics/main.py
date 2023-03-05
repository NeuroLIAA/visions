from . import constants
from .scripts.multimatch import Multimatch
from .scripts.human_scanpath_prediction import HumanScanpathPrediction
from .scripts.cumulative_performance import CumulativePerformance
from .scripts import utils
from os import path


def main(datasets, models, compute_cumulative_performance, compute_multimatch, compute_hsp, plot_results):
    datasets_results = {}
    for dataset_name in datasets:
        dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
        dataset_info = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))

        human_scanpaths_dir = path.join(dataset_path, dataset_info['scanpaths_dir'])
        dataset_results_dir = path.join(constants.RESULTS_PATH, dataset_name + '_dataset')
        if not path.isdir(dataset_results_dir):
            print('No results found for ' + dataset_name + ' dataset')
            continue

        max_scanpath_length = dataset_info['max_scanpath_length']
        # If desired, this number can be less than the total and the same random subset will be used for all models
        number_of_images = dataset_info['number_of_images']

        multimatch = Multimatch(dataset_name, human_scanpaths_dir, dataset_results_dir, number_of_images,
                                compute_multimatch)

        cumulative_performance = CumulativePerformance(dataset_name, number_of_images, max_scanpath_length,
                                                       compute_cumulative_performance)
        cumulative_performance.add_human_mean(human_scanpaths_dir, constants.HUMANS_COLOR)

        human_scanpath_prediction = HumanScanpathPrediction(dataset_name, human_scanpaths_dir, dataset_results_dir,
                                                            constants.MODELS_PATH, number_of_images, compute_hsp)

        color_index = 0
        models.sort(reverse=True)
        for model_name in models:
            if not path.isdir(path.join(dataset_results_dir, model_name)):
                print('No results found for ' + model_name + ' in ' + dataset_name + ' dataset')
                continue

            model_scanpaths_file = path.join(dataset_results_dir, model_name, 'Scanpaths.json')
            model_scanpaths = utils.load_dict_from_json(model_scanpaths_file)

            cumulative_performance.add_model(model_name, model_scanpaths, constants.MODELS_COLORS[color_index])

            # Human multimatch scores are different for each model, since each model uses different image sizes
            multimatch.load_human_mean_per_image(model_name, model_scanpaths)
            multimatch.add_model_vs_humans_mean_per_image(model_name, model_scanpaths,
                                                          constants.MODELS_COLORS[color_index])

            human_scanpath_prediction.compute_metrics_for_model(model_name)

            color_index += 1

        human_scanpath_prediction.add_baseline_models()

        cumulative_performance.save_results(save_path=dataset_results_dir, filename=constants.FILENAME)
        multimatch.save_results(save_path=dataset_results_dir, filename=constants.FILENAME)
        human_scanpath_prediction.save_results(save_path=dataset_results_dir, filename=constants.FILENAME)

        dataset_results = utils.load_dict_from_json(path.join(dataset_results_dir, constants.FILENAME))
        datasets_results[dataset_name] = dataset_results

        if plot_results:
            cumulative_performance.plot(save_path=dataset_results_dir)
            multimatch.plot(save_path=dataset_results_dir)
            utils.plot_table(utils.create_table(dataset_results), title=dataset_name + ' dataset',
                             save_path=dataset_results_dir, filename='Table.png')

    if compute_cumulative_performance and compute_multimatch and compute_hsp:
        final_table = utils.average_results(datasets_results, save_path=constants.RESULTS_PATH, filename='Scores.csv')
        if plot_results:
            utils.plot_table(final_table, title='Ranking', save_path=constants.RESULTS_PATH, filename='Ranking.png')
