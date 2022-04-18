from scipy.stats import multivariate_normal, gaussian_kde
from . import utils
from .. import constants
from os import path, listdir, pardir
import pandas as pd
import numpy as np
import shutil
import numba
import importlib

""" Computes Human Scanpath Prediction on the visual search models for a given dataset. 
    See "Kümmerer, M. & Bethge, M. (2021), State-of-the-Art in Human Scanpath Prediction" for more information.
    The methods for computing AUC and NSS were taken from https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py
"""

class HumanScanpathPrediction:
    def __init__(self, dataset_name, human_scanpaths_dir, dataset_results_dir, models_dir, number_of_images, compute):
        self.models_results      = {}
        self.dataset_name        = dataset_name
        self.number_of_images    = number_of_images
        self.human_scanpaths_dir = human_scanpaths_dir
        self.dataset_results_dir = dataset_results_dir
        self.models_dir          = models_dir

        self.null_object = not compute

    def compute_metrics_for_model(self, model_name):
        if self.null_object: return

        model_output_path     = path.join(self.dataset_results_dir, model_name)    
        human_scanpaths_files = utils.sorted_alphanumeric(listdir(self.human_scanpaths_dir))
        model_average_file    = path.join(model_output_path, 'human_scanpath_prediction_mean_per_image.json')

        average_results_per_image = utils.load_dict_from_json(model_average_file)
        if average_results_per_image:
            print('[Human Scanpath Prediction] Found previously computed results for ' + model_name)
        else:
            for subject in human_scanpaths_files:
                subject_number = subject[4:6]

                if not self.subject_already_processed(subject, subject_number, model_output_path):
                    model = importlib.import_module(self.models_dir + '.' + model_name + '.main')
                    print('[Human Scanpath Prediction] Running ' + model_name + ' on ' + self.dataset_name + ' dataset using subject ' + subject_number + ' scanpaths')
                    model.main(self.dataset_name, int(subject_number))
            
            average_results_per_image = self.average_results(model_output_path)
            utils.save_to_json(model_average_file, average_results_per_image)

        self.compute_model_mean(average_results_per_image, model_name)

    def subject_already_processed(self, subject_file, subject_number, model_output_path):
        subjects_predictions_path  = path.join(model_output_path, 'subjects_predictions')
        subject_scanpath_file      = path.join(self.human_scanpaths_dir, subject_file)
        subject_predictions_file   = path.join(subjects_predictions_path, 'subject_' + subject_number + '_results.json')
        if utils.is_contained_in(subject_scanpath_file, subject_predictions_file):
            print('[Human Scanpath Prediction] Found previously computed results for subject ' + subject_number)
            return True
        
        return False
    
    def compute_model_mean(self, average_results_per_image, model_name):
        """ Get the average across all images for a given model in a given dataset """
        self.models_results[model_name] = {'AUChsp': 0, 'NSShsp': 0, 'IGhsp': 0}

        number_of_images            = min(len(average_results_per_image), self.number_of_images)
        results_per_image_subsample = utils.get_random_subset(average_results_per_image, size=number_of_images)
        for image_name in results_per_image_subsample:
            self.models_results[model_name]['AUChsp'] += results_per_image_subsample[image_name]['AUC'] / number_of_images
            self.models_results[model_name]['NSShsp'] += results_per_image_subsample[image_name]['NSS'] / number_of_images
            self.models_results[model_name]['IGhsp'] += results_per_image_subsample[image_name]['IG'] / number_of_images
    
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
            # Round
            for metric in self.models_results[model]:
                self.models_results[model][metric] = np.round(self.models_results[model][metric], 3)
                
            utils.update_dict(dataset_metrics, model, self.models_results[model])

        utils.save_to_json(dataset_metrics_file, dataset_metrics)

def save_scanpath_prediction_metrics(subject_scanpath, image_name, output_path):
    """ After creating the probability maps for each fixation in a given human subject's scanpath, visual search models call this method """
    probability_maps_path = path.join(output_path, 'probability_maps', image_name[:-4])
    if not path.exists(probability_maps_path):
        print('[Human Scanpath Prediction] No probability maps found for ' + image_name)
        return
    probability_maps = listdir(probability_maps_path)
    dataset_name     = subject_scanpath['dataset'][:-8]

    subject_fixations_x = np.array(subject_scanpath['X'], dtype=int)
    subject_fixations_y = np.array(subject_scanpath['Y'], dtype=int)

    image_roc, image_nss, image_igs = [], [], []
    for index in range(1, len(probability_maps) + 1):
        probability_map = pd.read_csv(path.join(probability_maps_path, 'fixation_' + str(index) + '.csv')).to_numpy()
        auc, nss, ig    = compute_metrics(probability_map, subject_fixations_y[index], subject_fixations_x[index], dataset_name, output_path)
        image_roc.append(auc)
        image_nss.append(nss)
        image_igs.append(ig)

    subject   = path.basename(output_path)
    file_path = path.join(output_path, pardir, subject + '_results.json')
    if path.exists(file_path):
        model_subject_metrics = utils.load_dict_from_json(file_path)
    else:
        model_subject_metrics = {}
    
    model_subject_metrics[image_name] = {'AUC': np.mean(image_roc), 'NSS': np.mean(image_nss), 'IG': np.mean(image_igs)}  
    utils.save_to_json(file_path, model_subject_metrics)

    # Clean up probability maps if their size is too big
    if utils.dir_is_too_heavy(probability_maps_path):
        shutil.rmtree(probability_maps_path)

def compute_metrics(probability_map, human_fixation_y, human_fixation_x, dataset_name, output_path):
    baseline_map = center_bias(probability_map.shape, dataset_name, output_path)
    # baseline_map = center_gaussian(probability_map.shape)

    # import matplotlib.pyplot as plt
    # plt.imshow(baseline_map)
    # plt.colorbar()
    # plt.show()

    auc = AUC(probability_map, human_fixation_y, human_fixation_x)
    nss = NSS(probability_map, human_fixation_y, human_fixation_x)
    ig  = infogain(probability_map, baseline_map, human_fixation_y, human_fixation_x)

    return auc, nss, ig

def center_bias(shape, dataset_name, output_path):
    filepath = path.join(output_path, pardir, 'center_bias.csv')
    if path.exists(filepath):
        return pd.read_csv(filepath).to_numpy()

    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
    dataset_info = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))
    human_scanpaths_dir = path.join(dataset_path, dataset_info['scanpaths_dir'])

    scanpaths_X = []
    scanpaths_Y = []
    for subject_file in listdir(human_scanpaths_dir):
        subject_scanpaths = utils.load_dict_from_json(path.join(human_scanpaths_dir, subject_file))
        for image_name in subject_scanpaths:
            trial = subject_scanpaths[image_name]
            trial_scanpath_X = [utils.rescale_coordinate(x, trial['image_width'], shape[1]) for x in trial['X']]
            trial_scanpath_Y = [utils.rescale_coordinate(y, trial['image_height'], shape[0]) for y in trial['Y']]

            scanpaths_X += trial_scanpath_X
            scanpaths_Y += trial_scanpath_Y

    scanpaths_X = np.array(scanpaths_X)
    scanpaths_Y = np.array(scanpaths_Y)

    xmin, xmax = scanpaths_X.min(), scanpaths_X.max()
    ymin, ymax = scanpaths_Y.min(), scanpaths_Y.max()
    X, Y = np.mgrid[ymin:ymax:(shape[0] * 1j), xmin:xmax:(shape[1] * 1j)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([scanpaths_Y, scanpaths_X])
    kernel = gaussian_kde(values)
    centerbias = np.reshape(kernel(positions).T, X.shape)

    utils.save_to_csv(centerbias, filepath)

    return centerbias

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

def NSS(probability_map, ground_truth_fixation_y, ground_truth_fixation_x):
    """ The returned array has length equal to the number of fixations """
    mean  = np.mean(probability_map)
    std   = np.std(probability_map)
    value = np.copy(probability_map[ground_truth_fixation_y, ground_truth_fixation_x])
    value -= mean

    if std:
        value /= std

    return value

def infogain(s_map, baseline_map, ground_truth_fixation_y, ground_truth_fixation_x):
    eps = 2.2204e-16

    s_map        = s_map / np.sum(s_map)
    baseline_map = baseline_map / np.sum(baseline_map)
    
    return np.log2(eps + s_map[ground_truth_fixation_y, ground_truth_fixation_x]) - np.log2(eps + baseline_map[ground_truth_fixation_y, ground_truth_fixation_x])

def AUC(probability_map, ground_truth_fixation_y, ground_truth_fixation_x):
    """ Calculate AUC score for a given fixation """
    positive  = probability_map[ground_truth_fixation_y, ground_truth_fixation_x]
    negatives = probability_map.flatten()

    return auc_for_one_positive(positive, negatives)

def auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.
    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    return _auc_for_one_positive(positive, np.asarray(negatives))

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