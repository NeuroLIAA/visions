from os import path, listdir
import shutil
import constants

def delete_precomputed_results(dataset_name, model_name):
    model_results = path.join(path.join(constants.RESULTS_PATH, dataset_name + '_dataset'), model_name)
    shutil.rmtree(model_results)

def found_precomputed_results(dataset_name, model_name):
    results_file = path.join(path.join(path.join(constants.RESULTS_PATH, dataset_name + '_dataset'), model_name), 'Scanpaths.json')
    return path.exists(results_file)

def get_dirs(path_):
    files = listdir(path_)
    dirs  = [dir_ for dir_ in files if path.isdir(path.join(path_, dir_))]

    return dirs

def get_files(path_):
    files = listdir(path_)
    files = [file.split('.')[0] for file in files if path.isfile(path.join(path_, file))]

    return files