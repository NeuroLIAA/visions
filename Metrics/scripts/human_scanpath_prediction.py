from os import listdir, path
from tqdm import tqdm
from scipy.stats import multivariate_normal
from . import utils
from os import path, listdir, pardir, scandir
import pandas as pd
import numpy as np
import shutil
import numba
import importlib

""" Computes human scanpath prediction on the visual search models for a given dataset. 
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
        if self.null_object: return

        model_output_path     = path.join(self.dataset_results_dir, model_name)    
        human_scanpaths_files = utils.sorted_alphanumeric(listdir(self.human_scanpaths_dir))
        model_average_file    = path.join(model_output_path, 'scanpath_prediction_mean_per_image.json')

        average_results_per_image = utils.load_dict_from_json(model_average_file)
        if average_results_per_image:
            print('[Scanpath prediction] Found previously computed results for ' + model_name)
        else:
            for subject in human_scanpaths_files:
                subject_number = subject[4:6]
                subjects_predictions_path  = path.join(model_output_path, 'subjects_predictions')
                if path.exists(subjects_predictions_path) and 'subject_' + subject_number + '_results.json' in utils.list_json_files(subjects_predictions_path):
                    print('[Scanpath prediction] Found previously computed results for subject ' + subject_number)
                    continue

                model = importlib.import_module(self.models_dir + '.' + model_name + '.main')
                print('[Scanpath prediction] Running ' + model_name + ' on ' + self.dataset_name + ' dataset using subject ' + subject_number + ' scanpaths')
                model.main(self.dataset_name, int(subject_number))
            
            average_results_per_image = self.average_results(model_output_path)
            utils.save_to_json(model_average_file, average_results_per_image)

        self.compute_model_mean(average_results_per_image, model_name)
    
    def compute_model_mean(self, average_results_per_image, model_name):
        """ Get the average across all images for a given model in a given dataset """
        self.models_results[model_name] = {'Scanpath_prediction': {'AUC': 0, 'NSS': 0, 'IG': 0}}
        number_of_images = len(average_results_per_image)
        for image_name in average_results_per_image:
            self.models_results[model_name]['Scanpath_prediction']['AUC'] += average_results_per_image[image_name]['AUC'] / number_of_images
            self.models_results[model_name]['Scanpath_prediction']['NSS'] += average_results_per_image[image_name]['NSS'] / number_of_images
            self.models_results[model_name]['Scanpath_prediction']['IG']  += average_results_per_image[image_name]['IG'] / number_of_images
    
    def average_results(self, model_output_path):
        """ Get the average of all subjects for each image """
        subjects_results_path  = path.join(model_output_path, 'subjects_predictions')
        subjects_results_files = utils.list_json_files(subjects_results_path)
        average_results_per_image   = {}
        number_of_results_per_image = {}
        # Sum image values across subjects
        for subject_file in subjects_results_files:
            subject_results = utils.load_dict_from_json(path.join(subjects_results_path, subject_file))
            for image_name in subject_results:
                metrics = subject_results[image_name]
                if image_name in average_results_per_image:
                    average_results_per_image[image_name]['AUC'] += metrics['AUC']
                    average_results_per_image[image_name]['NSS'] += metrics['NSS']
                    average_results_per_image[image_name]['IG']  += metrics['IG']
                    number_of_results_per_image[image_name] += 1
                else:
                    average_results_per_image[image_name]   = metrics
                    number_of_results_per_image[image_name] = 1
        
        # Average image values across subjects
        for image_name in average_results_per_image:
            average_results_per_image[image_name]['AUC'] /= number_of_results_per_image[image_name]
            average_results_per_image[image_name]['NSS'] /= number_of_results_per_image[image_name]
            average_results_per_image[image_name]['IG']  /= number_of_results_per_image[image_name]

        return average_results_per_image

    def save_results(self, save_path, filename):
        if self.null_object: return

        dataset_metrics_file = path.join(save_path, filename)
        dataset_metrics      = utils.load_dict_from_json(dataset_metrics_file)
    
        for model in self.models_results:
            utils.update_dict(dataset_metrics, model, self.models_results[model])

        utils.save_to_json(dataset_metrics_file, dataset_metrics)

def save_scanpath_prediction_metrics(subject_scanpath, image_name, output_path):
    """ After creating the probability maps for each fixation in a given human subject's scanpath, visual search models call this method """
    probability_maps_path = path.join(output_path, path.join('probability_maps', image_name[:-4]))
    if not path.exists(probability_maps_path):
        print('[Scanpath prediction] No probability maps found for ' + image_name)
        return
    probability_maps = listdir(probability_maps_path)

    subject_fixations_x = np.array(subject_scanpath['X'], dtype=int)
    subject_fixations_y = np.array(subject_scanpath['Y'], dtype=int)

    image_rocs, image_nss, image_igs = [], [], []
    for index in range(1, len(probability_maps) + 1):
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

    # Clean up probability maps if their size is too big
    if utils.dir_is_too_heavy(probability_maps_path):
        shutil.rmtree(probability_maps_path)

def compute_metrics(probability_map, human_fixations_y, human_fixations_x):
    probability_map = probability_map.to_numpy(dtype=np.float)
    probability_map = normalize(probability_map)
    baseline_map    = center_gaussian(probability_map.shape)

    roc = np.mean(AUCs(probability_map, human_fixations_y, human_fixations_x)) # ¿Promediamos?
    nss = np.mean(NSS(probability_map, human_fixations_y, human_fixations_x)) # ¿Promediamos? 
    ig  = np.mean(infogain(probability_map, baseline_map, human_fixations_y, human_fixations_x)) # ¿Promediamos? 

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

    return temp

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