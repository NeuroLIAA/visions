from constants import MODELS_PATH, DATASETS_PATH, AVAILABLE_METRICS
import argparse
from utils import delete_precomputed_results, precomputed_results, get_dirs
from importlib import import_module
import Metrics.main as run_metrics


def main(datasets, models, metrics, plot_results, force_execution):
    for dataset_name in datasets:
        for model_name in models:
            if force_execution:
                delete_precomputed_results(dataset_name, model_name)

            if precomputed_results(dataset_name, model_name):
                print(f'Found precomputed results for {model_name} in {dataset_name} dataset')
                continue

            print(f'Running {model_name} on {dataset_name} dataset')
            model = import_module(f'{MODELS_PATH}.{model_name}.main')
            model.main(dataset_name)

    if metrics:
        cum_perf = 'perf' in metrics
        multimatch = 'mm' in metrics
        human_scanpath_prediction = 'hsp' in metrics

        run_metrics.main(datasets, models, cum_perf, multimatch, human_scanpath_prediction, plot_results)


if __name__ == "__main__":
    available_models, available_datasets = get_dirs(MODELS_PATH), get_dirs(DATASETS_PATH)
    parser = argparse.ArgumentParser(
        description='Run a given set of visual search models on specific datasets and compute the corresponding metrics'
    )
    parser.add_argument('--d', '--datasets', type=str, nargs='*', default=available_datasets,
                        help=f'Names of the datasets on which to run the models. \
        Values must be in list: {available_datasets}')
    parser.add_argument('--m', '--models', type=str, nargs='*', default=available_models,
                        help=f'Names of the models to run. \
        Values must be in list: {available_models}')
    parser.add_argument('--mts', '--metrics', type=str, nargs='*', default=AVAILABLE_METRICS,
                        help=f'Names of the metrics to compute. \
        Values must be in list: {AVAILABLE_METRICS}. \
        Leave blank to not run any. WARNING: If not precomputed, human scanpath prediction (hsp) will take a LONG time!'
                        )
    parser.add_argument('--noplot', action='store_true',
                        help='Do not plot metrics. Useful for leaving it running and going AFK.')
    parser.add_argument('--f', '--force', action='store_true',
                        help='Deletes all precomputed results and forces models\' execution.')

    args = parser.parse_args()
    invalid_models = not all(model in available_models for model in args.m)
    invalid_datasets = not all(dataset in available_datasets for dataset in args.d)
    invalid_metrics = not all(metric in AVAILABLE_METRICS for metric in args.mts)
    if (not args.m or invalid_models) or (not args.d or invalid_datasets) or invalid_metrics:
        raise ValueError('Invalid set of models, datasets or metrics')

    main(args.d, args.m, args.mts, not args.noplot, args.f)
