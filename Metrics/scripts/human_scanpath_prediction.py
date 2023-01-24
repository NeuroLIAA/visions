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
            
            average_results_per_image = self.get_model_average_per_image(model_output_path)
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
        """ Get the average scores across all images for a given model in a given dataset """
        self.models_results[model_name] = {'AUChsp': 0, 'NSShsp': 0, 'IGhsp': 0, 'LLhsp': 0}

        number_of_images            = min(len(average_results_per_image), self.number_of_images)
        results_per_image_subsample = utils.get_random_subset(average_results_per_image, size=number_of_images)
        for image_name in results_per_image_subsample:
            for metric in results_per_image_subsample[image_name]:
                self.models_results[model_name][metric + 'hsp'] += results_per_image_subsample[image_name][metric] / number_of_images

    def get_model_average_per_image(self, model_output_path):
        subjects_results_path  = path.join(model_output_path, 'subjects_predictions')
        subjects_results_files = utils.list_json_files(subjects_results_path)

        average_per_image = self.get_average_per_image(subjects_results_files, load_from_file=True, filespath=subjects_results_path)

        return average_per_image

    def get_average_per_image(self, subjects_results, load_from_file=False, filespath=None):
        """ Get the average score for each image of all subjects for a given model """
        average_per_image = {}
        number_of_results_per_image  = {}
        for subject in subjects_results:
            if load_from_file:
                subject_results = utils.load_dict_from_json(path.join(filespath, subject))
            else:
                subject_results = subjects_results[subject]

            for image_name in subject_results:
                trial_results = subject_results[image_name]
                if image_name in average_per_image:
                    for metric in average_per_image[image_name]:
                        average_per_image[image_name][metric] += trial_results[metric]
                    number_of_results_per_image[image_name] += 1
                else:
                    average_per_image[image_name] = trial_results
                    number_of_results_per_image[image_name] = 1

        for image_name in average_per_image:
            for metric in average_per_image[image_name]:
                average_per_image[image_name][metric] /= number_of_results_per_image[image_name]
        
        return average_per_image

    def add_baseline_models(self):
        baseline_filepath = path.join(self.dataset_results_dir, 'baseline_hsp.json')
        baseline_averages = utils.load_dict_from_json(baseline_filepath)
        if not baseline_averages:
            baseline_averages = self.run_baseline_models(baseline_filepath)

        for model in baseline_averages:
            self.compute_model_mean(baseline_averages[model], model)

    def run_baseline_models(self, filepath):
        """ Compute every metric for center bias, uniform and gold standard models in the given dataset """
        dataset_path = path.join(constants.DATASETS_PATH, self.dataset_name)
        dataset_info = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))
        image_size   = (dataset_info['image_height'], dataset_info['image_width'])

        baseline_models = {'center_bias': {'probability_map': center_bias(shape=image_size), 'results': {}}, \
                            'uniform': {'probability_map': uniform(shape=image_size), 'results': {}},
                            'gold_standard': {'probability_map': None, 'results': {}}}

        subjects_scanpaths_path  = path.join(dataset_path, dataset_info['scanpaths_dir'])
        subjects_scanpaths_files = utils.sorted_alphanumeric(listdir(subjects_scanpaths_path))
        for subject_scanpaths_file in subjects_scanpaths_files:
            subject = subject_scanpaths_file[:-15]
            print('[Human Scanpath Prediction] Running baseline models on ' + self.dataset_name + ' dataset using ' + subject + ' scanpaths')
            subject_scanpaths = utils.load_dict_from_json(path.join(subjects_scanpaths_path, subject_scanpaths_file))
            for image_name in subject_scanpaths:
                gold_standard_model = gold_standard(image_name, image_size, subjects_scanpaths_path, excluded_subject=subject)
                baseline_models['gold_standard']['probability_map'] = gold_standard_model

                trial_info = subject_scanpaths[image_name]
                scanpath_x = [int(x) for x in trial_info['X']]
                scanpath_y = [int(y) for y in trial_info['Y']]

                for model in baseline_models:
                    probability_map = baseline_models[model]['probability_map']
                    if probability_map is not None:
                        trial_aucs, trial_nss, trial_igs, trial_lls = compute_trial_metrics(len(scanpath_x), scanpath_x, scanpath_y, \
                            prob_maps_path=None, baseline_map=probability_map)
                        
                        model_results = baseline_models[model]['results']
                        if subject in model_results:
                            model_results[subject][image_name] = {'AUC': np.mean(trial_aucs), 'NSS': np.mean(trial_nss), 'IG': np.mean(trial_igs), 'LL': np.mean(trial_lls)}
                        else:
                            model_results[subject] = {image_name: {'AUC': np.mean(trial_aucs), 'NSS': np.mean(trial_nss), 'IG': np.mean(trial_igs), 'LL': np.mean(trial_lls)}}

        baseline_averages = {}
        for model in baseline_models:
            baseline_averages[model] = self.get_average_per_image(baseline_models[model]['results'])

        utils.save_to_json(filepath, baseline_averages)

        return baseline_averages

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

    subject_fixations_x = np.array(subject_scanpath['X'], dtype=int)
    subject_fixations_y = np.array(subject_scanpath['Y'], dtype=int)

    trial_aucs, trial_nss, trial_igs, trial_lls = compute_trial_metrics(len(probability_maps) + 1, subject_fixations_x, subject_fixations_y, probability_maps_path)

    subject   = path.basename(output_path)
    file_path = path.join(output_path, pardir, subject + '_results.json')
    if path.exists(file_path):
        model_subject_metrics = utils.load_dict_from_json(file_path)
    else:
        model_subject_metrics = {}
    
    model_subject_metrics[image_name] = {'AUC': np.mean(trial_aucs), 'NSS': np.mean(trial_nss), 'IG': np.mean(trial_igs), 'LL': np.mean(trial_lls)}  
    utils.save_to_json(file_path, model_subject_metrics)

    # Clean up probability maps if their size is too big
    if utils.dir_is_too_heavy(probability_maps_path):
        shutil.rmtree(probability_maps_path)

def compute_trial_metrics(number_of_fixations, subject_fixations_x, subject_fixations_y, prob_maps_path, baseline_map=None):
    trial_aucs, trial_nss, trial_igs, trial_lls = [], [], [], []
    for index in range(1, number_of_fixations):
        if baseline_map is None:
            fixation_prob_map = pd.read_csv(path.join(prob_maps_path, 'fixation_' + str(index) + '.csv')).to_numpy()
        else:
            fixation_prob_map = baseline_map
        baseline_ig      = center_bias(fixation_prob_map.shape)
        baseline_ll      = uniform(fixation_prob_map.shape)
        auc, nss, ig, ll = compute_fixation_metrics(baseline_ig, baseline_ll, fixation_prob_map, subject_fixations_y[index], subject_fixations_x[index])
        trial_aucs.append(auc)
        trial_nss.append(nss)
        trial_igs.append(ig)
        trial_lls.append(ll)
    
    return trial_aucs, trial_nss, trial_igs, trial_lls

def compute_fixation_metrics(baseline_ig, baseline_ll, probability_map, human_fixation_y, human_fixation_x):
    auc = AUC(probability_map, human_fixation_y, human_fixation_x)
    nss = NSS(probability_map, human_fixation_y, human_fixation_x)
    ig  = infogain(probability_map, baseline_ig, human_fixation_y, human_fixation_x)
    ll  = infogain(probability_map, baseline_ll, human_fixation_y, human_fixation_x)

    return auc, nss, ig, ll

def uniform(shape):
    return np.ones(shape) / (shape[0] * shape[1])

def center_bias(shape):
    shape_dir = str(shape[0]) + 'x' + str(shape[1])
    filepath  = path.join(constants.CENTER_BIAS_PATH, shape_dir,  'center_bias.pkl')
    if path.exists(filepath):
        return utils.load_pickle(filepath)

    scanpaths_X, scanpaths_Y = utils.load_center_bias_fixations(model_size=shape)
    centerbias = utils.gaussian_kde(scanpaths_X, scanpaths_Y, shape)

    utils.save_to_pickle(centerbias, filepath)

    return centerbias

def gold_standard(image_name, image_size, subjects_scanpaths_path, excluded_subject):    
    scanpaths_X, scanpaths_Y = utils.aggregate_scanpaths(subjects_scanpaths_path, image_name, excluded_subject)
    if len(scanpaths_X) == 0:
        goldstandard_model = None
    else:
        goldstandard_model = utils.gaussian_kde(scanpaths_X, scanpaths_Y, image_size)

    return goldstandard_model

def NSS(probability_map, ground_truth_fixation_y, ground_truth_fixation_x, eps=2.2204e-20):
    """ The returned array has length equal to the number of fixations """
    mean  = np.mean(probability_map)
    std   = np.std(probability_map)
    value = np.copy(probability_map[ground_truth_fixation_y, ground_truth_fixation_x])
    value -= mean
    value = value if eps < value else 0.0

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