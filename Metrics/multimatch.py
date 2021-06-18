import multimatch_gaze as mm
import json
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path

class Multimatch:
    def __init__(self, dataset_name, human_scanpaths_dir, dataset_results_dir):
        self.models_vs_humans_multimatch = []
        self.humans_multimatch = {}
        self.dataset_name = dataset_name
        self.human_scanpaths_dir = human_scanpaths_dir
        self.dataset_results_dir = dataset_results_dir

    def plot(self, save_path):
        number_of_models = len(self.models_vs_humans_multimatch)
        fig, axs = plt.subplots(1, number_of_models)

        ax_index = 0
        for model in self.models_vs_humans_multimatch:
            model_name = model['model_name']
            model_vs_human_multimatch = model['results']
            self.add_to_plot(axs[ax_index], model_name, model_vs_human_multimatch, self.humans_multimatch)
            ax_index += 1

        min_x, max_x = (1, 0)
        min_y, max_y = (1, 0)
        for ax in axs:
            ax.set(xlabel='Model vs human multimatch mean', ylabel='Human multimatch mean')
            ax.label_outer()
            min_x, max_x = min(min(ax.get_xlim()), min_x), max(max(ax.get_xlim()), max_x)
            min_y, max_y = min(min(ax.get_ylim()), min_y), max(max(ax.get_ylim()), max_y)

        fig.suptitle(self.dataset_name + ' dataset')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.savefig(save_path + 'Multimatch against humans.png')    
        plt.show()

    def add_to_plot(self, ax, model_name, multimatch_values_per_image_x, multimatch_values_per_image_y):
        x_vector = []
        y_vector = []
        trials_names = []
        for image_name in multimatch_values_per_image_x.keys():
            if not(image_name in multimatch_values_per_image_y):
                continue

            # Exclude temporal dimension and compute the mean of all the others dimensions
            value_x = np.mean(multimatch_values_per_image_x[image_name][:-1])
            value_y = np.mean(multimatch_values_per_image_y[image_name][:-1])

            trials_names.append(image_name)

            x_vector.append(value_x)
            y_vector.append(value_y)

        # Plot multimatch
        ax.scatter(x_vector, y_vector, color='red', alpha=0.5)
        # Plot y = x function
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, linestyle='dashed', c='.3')
        # Plot linear regression
        x_linear   = np.array(x_vector)[:, np.newaxis]
        m, _, _, _ = np.linalg.lstsq(x_linear, y_vector, rcond=None)
        ax.plot(x_vector, m * x_linear, linestyle=(0, (5, 5)), color='purple', alpha=0.5)

        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
        ax.set_title(model_name)

        # Get most-similar to less-similar trials names
        scores_diff = np.array(x_vector) - np.array(y_vector)
        self.print_trials_names_in_similarity_order(scores_diff, trials_names)

    def print_trials_names_in_similarity_order(self, scores_diff, trials_names):
        scores_diff = list(zip(scores_diff, trials_names))
        scores_right_half = []
        scores_left_half  = []
        for trial in scores_diff:
            if trial[0] > 0:
                scores_right_half.append(trial)
            else:
                scores_left_half.append((abs(trial[0]), trial[1]))

        scores_right_half.sort(key=lambda elem: elem[0])
        scores_left_half.sort(key=lambda elem: elem[0])
        print('Dataset: ' + self.dataset_name + '. Model: ' + model_name + '. Most similar to less similar trials')
        print('Right half:')
        print(scores_right_half)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Left half:')
        print(scores_left_half)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def add_model_vs_humans_mean_per_image(self, model_name, model_scanpaths):
        " For each scanpath produced by the model, multimatch is calculated against the scanpath of that same image for every human subject "
        " The mean is computed for each image "
        " Output is a dictionary where the image names are the keys and the multimatch means are the values "
        multimatch_model_vs_humans_mean_per_image = {}

        total_values_per_image = {}
        subjects_scanpaths_files = listdir(self.human_scanpaths_dir)
        for subject_filename in subjects_scanpaths_files:
            with open(self.human_scanpaths_dir + subject_filename, 'r') as fp:
                subject_scanpaths = json.load(fp)
            for image_name in model_scanpaths.keys():
                if not(image_name in subject_scanpaths):
                    continue

                model_trial_info = model_scanpaths[image_name]
                subject_trial_info = subject_scanpaths[image_name]

                trial_multimatch_result = self.compute_multimatch(subject_trial_info, model_trial_info)

                # Check if result is empty
                if not trial_multimatch_result:
                    continue

                if image_name in multimatch_model_vs_humans_mean_per_image:
                    multimatch_trial_value_acum = multimatch_model_vs_humans_mean_per_image[image_name]
                    multimatch_model_vs_humans_mean_per_image[image_name] = np.add(multimatch_trial_value_acum, trial_multimatch_result)
                    total_values_per_image[image_name] += 1 
                else:
                    multimatch_model_vs_humans_mean_per_image[image_name] = trial_multimatch_result
                    total_values_per_image[image_name] = 1

        # Compute mean per image
        for image_name in multimatch_model_vs_humans_mean_per_image.keys():
            multimatch_model_vs_humans_mean_per_image[image_name] = (np.divide(multimatch_model_vs_humans_mean_per_image[image_name], total_values_per_image[image_name])).tolist()
        
        self.models_vs_humans_multimatch.append({'model_name': model_name, 'results': multimatch_model_vs_humans_mean_per_image})

    def load_human_mean_per_image(self):
        " For each human subject, multimatch is computed against every other human subject, for each trial "
        " The mean is computed for each trial (i.e. for each image) "
        " Output is a dictionary where the image names are the keys and the multimatch means are the values "
        multimatch_human_mean_per_image = {}
        # Check if it was already computed
        multimatch_human_mean_json_file = self.dataset_results_dir + 'multimatch_human_mean_per_image.json'
        if path.exists(multimatch_human_mean_json_file):
            with open(multimatch_human_mean_json_file, 'r') as fp:
                multimatch_human_mean_per_image = json.load(fp)
        else:
            total_values_per_image = {}
            # Compute multimatch for each image for every pair of subjects
            subjects_scanpaths_files = listdir(self.human_scanpaths_dir)
            for subject_filename in list(subjects_scanpaths_files):
                subjects_scanpaths_files.remove(subject_filename) 
                with open(self.human_scanpaths_dir + subject_filename, 'r') as fp:
                    subject_scanpaths = json.load(fp)
                for subject_to_compare_filename in subjects_scanpaths_files:
                    with open(self.human_scanpaths_dir + subject_to_compare_filename, 'r') as fp:
                        subject_to_compare_scanpaths = json.load(fp)
                    for image_name in subject_scanpaths.keys():
                        if not(image_name in subject_to_compare_scanpaths):
                            continue

                        subject_trial_info = subject_scanpaths[image_name]
                        subject_to_compare_trial_info = subject_to_compare_scanpaths[image_name]

                        trial_multimatch_result = self.compute_multimatch(subject_trial_info, subject_to_compare_trial_info)

                        # Check if result is empty
                        if not trial_multimatch_result:
                            continue

                        if image_name in multimatch_human_mean_per_image:
                            multimatch_trial_value_acum = multimatch_human_mean_per_image[image_name]
                            multimatch_human_mean_per_image[image_name] = np.add(multimatch_trial_value_acum, trial_multimatch_result)
                            total_values_per_image[image_name] += 1 
                        else:
                            multimatch_human_mean_per_image[image_name] = trial_multimatch_result
                            total_values_per_image[image_name] = 1

            # Compute mean per image
            for image_name in multimatch_human_mean_per_image.keys():
                multimatch_human_mean_per_image[image_name] = (np.divide(multimatch_human_mean_per_image[image_name], total_values_per_image[image_name])).tolist()
            
            with open(multimatch_human_mean_json_file, 'w') as fp:
                json.dump(multimatch_human_mean_per_image, fp, indent = 4)

        self.humans_multimatch = multimatch_human_mean_per_image

    def compute_multimatch(self, trial_info, trial_to_compare_info):
        target_found = trial_info['target_found'] and trial_to_compare_info['target_found']
        if not(target_found):
            return []

        screen_size = [trial_info['image_width'], trial_info['image_height']]

        trial_scanpath_X = trial_info['X']
        trial_scanpath_Y = trial_info['Y']
        trial_scanpath_length = len(trial_scanpath_X)
        trial_scanpath_time = self.get_scanpath_time(trial_info, trial_scanpath_length)

        trial_to_compare_image_width  = trial_to_compare_info['image_width']
        trial_to_compare_image_height = trial_to_compare_info['image_height']

        trial_to_compare_scanpath_X = trial_to_compare_info['X']
        trial_to_compare_scanpath_Y = trial_to_compare_info['Y']
        trial_to_compare_scanpath_length = len(trial_to_compare_scanpath_X)
        trial_to_compare_scanpath_time = self.get_scanpath_time(trial_to_compare_info, trial_to_compare_scanpath_length)

        # Rescale accordingly
        trial_to_compare_scanpath_X = [self.rescale_coordinate(x, trial_to_compare_image_width, screen_size[0]) for x in trial_to_compare_scanpath_X]
        trial_to_compare_scanpath_Y = [self.rescale_coordinate(y, trial_to_compare_image_height, screen_size[1]) for y in trial_to_compare_scanpath_Y]

        # Multimatch can't be computed for scanpaths with length shorter than 3
        if trial_scanpath_length < 3 or trial_to_compare_scanpath_length < 3:
            return []

        trial_scanpath = np.array(list(zip(trial_scanpath_X, trial_scanpath_Y, trial_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
        trial_to_compare_scanpath = np.array(list(zip(trial_to_compare_scanpath_X, trial_to_compare_scanpath_Y, trial_to_compare_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])

        return mm.docomparison(trial_scanpath, trial_to_compare_scanpath, screen_size)

    def get_scanpath_time(self, trial_info, length):
        if 'T' in trial_info:
            scanpath_time = [t * 0.0001 for t in trial_info['T']]
        else:
            # Dummy
            scanpath_time = [0.3] * length
        
        return scanpath_time

    def rescale_coordinate(self, value, old_size, new_size):
        return (value / old_size) * new_size
