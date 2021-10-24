from os import listdir, path
from tqdm import tqdm
from scipy.stats import multivariate_normal
from . import utils
from os import path, listdir, pardir
import pandas as pd
import numpy as np
import shutil
import numba
import importlib

""" Computes human scanpath prediction on the visual search models for a given set of datasets. 
    See "Kümmerer, M. & Bethge, M. (2021), State-of-the-Art in Human Scanpath Prediction" for more information.
    The methods for computing AUC and NSS were taken from https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py
"""

class HumanScanpathPrediction:
    def __init__(self, dataset_name, human_scanpaths_dir, dataset_results_dir, models_dir, compute):
        self.models_results = {}
        self.dataset_name   = dataset_name
        self.human_scanpaths_dir = human_scanpaths_dir
        self.dataset_results_dir = dataset_results_dir
        self.models_dir          = models_dir

        self.null_object = not compute

    def compute_metrics_for_model(self, model_name):
        if self.null_object:
            return
            
        human_scanpaths_files = utils.sorted_alphanumeric(listdir(self.human_scanpaths_dir))
        model_output_path     = path.join(self.dataset_results_dir, model_name)
        for subject in human_scanpaths_files:
            subject_number  = subject[4:6]

            model = importlib.import_module(self.models_dir + '.' + model_name + '.main')
            print('Running ' + model_name + ' using subject ' + subject_number + ' scanpaths')
            model.main(self.dataset_name, int(subject_number))

            # TODO: Levantar los resultados del sujeto sobre todo el dataset y promediarlos


def save_scanpath_prediction_metrics(subject_scanpath, scanpath_length, image_name, output_path):
    """ After creating the probability maps for each fixation in a given human subject's scanpath, visual search models call this method """
    probability_maps_path = path.join(output_path, path.join('probability_maps', image_name[:-4]))
    probability_maps = listdir(probability_maps_path)

    subject_fixations_x = np.array(subject_scanpath['X'], dtype=int)
    subject_fixations_y = np.array(subject_scanpath['Y'], dtype=int)

    image_rocs, image_nss, image_igs = [], [], []
    # Since the model may have found the target earlier due to rescaling, its scanpath length is used.
    for index in range(1, scanpath_length):
        probability_map = pd.read_csv(path.join(probability_maps_path, 'fixation_' + str(index) + '.csv'))
        roc, nss, ig = compute_metrics(probability_map, subject_fixations_y[:index], subject_fixations_x[:index])
        image_rocs.append(roc)
        image_nss.append(nss)
        image_igs.append(ig)

    subject   = path.basename(output_path)
    file_path = path.join(path.join(output_path, pardir), subject + '_results.json')
    if path.exists(file_path):
        model_subject_metrics = utils.load_dict_from_json(file_path)
    else:
        model_subject_metrics = {}
    
    model_subject_metrics[image_name] = {'AUC': np.mean(image_rocs), 'NSS': np.mean(image_nss), 'IG': np.mean(image_igs)}  
    utils.save_to_json(file_path, model_subject_metrics)

    # Clean up
    shutil.rmtree(probability_maps_path)

def compute_metrics(probability_map, human_fixations_y, human_fixations_x):
    probability_map = probability_map.to_numpy(dtype=np.float)
    probability_map = normalize(probability_map)
    baseline_map    = center_gaussian(probability_map.shape)

    roc = np.mean(AUCs(probability_map, human_fixations_y, human_fixations_x)) # ¿Promediamos?
    nss = np.mean(NSS(probability_map, human_fixations_y, human_fixations_x)) # ¿Promediamos? 
    ig  = infogain(probability_map, baseline_map, human_fixations_y, human_fixations_x)

    return roc, nss, ig

def center_gaussian(shape):
    sigma  = [[1, 0], [0, 1]]
    mean   = [shape[0] // 2, shape[1] // 2]
    x_range = np.linspace(0, shape[0], shape[0])
    y_range = np.linspace(0, shape[1], shape[1])

    x_matrix, y_matrix = np.meshgrid(y_range, x_range)
    quantiles = np.transpose([y_matrix.flatten(), x_matrix.flatten()])
    mvn = multivariate_normal.pdf(quantiles, mean=mean, cov=sigma)
    mvn = np.reshape(mvn, shape)

    return mvn

def normalize(probability_map):
    normalized_probability_map = probability_map - np.min(probability_map)
    normalized_probability_map = normalized_probability_map / np.max(normalized_probability_map)

    return normalized_probability_map

def NSS(saliency_map, ground_truth_fixations_y, ground_truth_fixations_x):
    """ The returned array has length equal to the number of fixations """
    mean = np.mean(saliency_map)
    std  = np.std(saliency_map)
    value = np.copy(saliency_map[ground_truth_fixations_y, ground_truth_fixations_x])
    value -= mean

    if std:
        value /= std

    return value

def infogain(s_map, baseline_map, ground_truth_fixations_y, ground_truth_fixations_x):
    eps = 2.2204e-16

    s_map        = s_map / (np.sum(s_map) * 1.0)
    baseline_map = baseline_map / (np.sum(baseline_map) * 1.0)

    temp = []
    for i in zip(ground_truth_fixations_x, ground_truth_fixations_y):
        temp.append(np.log2(eps + s_map[i[1], i[0]]) - np.log2(eps + baseline_map[i[1], i[0]]))

    return np.mean(temp)

def AUCs(probability_map, ground_truth_fixations_y, ground_truth_fixations_x):
    """ Calculate AUC scores for fixations """
    rocs_per_fixation = []

    for i in tqdm(range(len(ground_truth_fixations_x)), total=len(ground_truth_fixations_x)):
        positive  = probability_map[ground_truth_fixations_y[i], ground_truth_fixations_x[i]]
        negatives = probability_map.flatten()

        this_roc = auc_for_one_positive(positive, negatives)
        rocs_per_fixation.append(this_roc)

    return np.asarray(rocs_per_fixation)

def auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.
    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    return _auc_for_one_positive(positive, np.asarray(negatives))

@numba.jit(nopython=True)
def fill_fixation_map(fixation_map, fixations):
    """fixationmap: 2d array. fixations: Nx2 array of y, x positions"""
    for i in range(len(fixations)):
        fixation_y, fixation_x = fixations[i]
        fixation_map[int(fixation_y), int(fixation_x)] += 1

@numba.jit(nopython=True)
def _auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.
    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    count = 0
    for negative in negatives:
        if negative < positive:
            count += 1
        elif negative == positive:
            count += 0.5

    return count / len(negatives)