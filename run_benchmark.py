import constants
import argparse
import utils
import importlib
import Metrics.main as metrics_module
from os import path, listdir

def main(datasets, models, metrics, force_execution):
    for dataset_name in datasets:
        for model_name in models:
            if not force_execution and utils.found_precomputed_results(dataset_name, model_name):
                print('Found precomputed results for ' + model_name + ' on ' + dataset_name + ' dataset')
                continue
            print('Running ' + model_name + ' on ' + dataset_name + ' dataset')
            model = importlib.import_module(constants.MODELS_PATH + '.' + model_name + '.main')
            model.main(dataset_name)
    
    if metrics:
        cum_perf   = 'perf' in metrics
        multimatch = 'mm' in metrics
        human_scanpath_prediction = 'hsp' in metrics

        metrics_module.main(cum_perf, multimatch, human_scanpath_prediction)

if __name__ == "__main__":
    available_models   = utils.get_dirs(constants.MODELS_PATH)
    available_datasets = utils.get_dirs(constants.DATASETS_PATH)
    available_metrics  = constants.AVAILABLE_METRICS
    parser = argparse.ArgumentParser(description='Run a given set of visual search models on specific datasets and compute the corresponding metrics')
    parser.add_argument('--d', '--datasets', type=str, nargs='*', default=available_datasets, help='Names of the datasets on which to run the models. \
        Values must be in list: ' + str(available_datasets))
    parser.add_argument('--m', '--models', type=str, nargs='*', default=available_models, help='Names of the models to run. \
        Values must be in list: ' + str(available_models))
    parser.add_argument('--mts', '--metrics', type=str, nargs='*', default=available_metrics, help='Names of the metrics to compute. \
        Values must be in list: ' + str(available_metrics) + '. Leave blank to not run any. WARNING: If not precomputed, human scanpath prediction (hsp) will take a LONG time!')
    parser.add_argument('--f', '--force', action='store_true', help='Ignore precomputed results and force models\' execution.')

    args = parser.parse_args()
    invalid_models   = not all(model in available_models for model in args.m)
    invalid_datasets = not all(dataset in available_datasets for dataset in args.d)
    invalid_metrics  = not all(metric in available_metrics for metric in args.mts)
    if (not args.m or invalid_models) or (not args.d or invalid_datasets) or invalid_metrics:
        raise ValueError('Invalid set of models, datasets or metrics')

    main(args.d, args.m, args.mts, args.f)