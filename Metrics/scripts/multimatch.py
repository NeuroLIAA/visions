import multimatch_gaze as mm
import numpy as np
import matplotlib.pyplot as plt
from . import utils
from scipy.stats import pearsonr
from os import listdir, path

class Multimatch:
    def __init__(self, dataset_name, human_scanpaths_dir, dataset_results_dir, number_of_images, compute):
        self.multimatch_values = {}
        self.dataset_name = dataset_name
        self.human_scanpaths_dir = human_scanpaths_dir
        self.dataset_results_dir = dataset_results_dir
        self.number_of_images = number_of_images
        
        self.null_object = not compute

    def plot(self, save_path):
        if self.null_object:
            return
            
        number_of_models = len(self.multimatch_values)
        fig, axs = plt.subplots(1, number_of_models, sharex=True, sharey=True, figsize=(10, 5))

        if number_of_models == 1: axs = [axs]

        ax_index = 0
        for model in self.multimatch_values:
            model_name = model
            model_vs_human_multimatch = utils.get_random_subset(self.multimatch_values[model]['model_vs_humans'], size=self.number_of_images)
            humans_multimatch         = utils.get_random_subset(self.multimatch_values[model]['human_mean'], size=self.number_of_images)
            model_color               = self.multimatch_values[model]['plot_color']

            self.add_to_plot(axs[ax_index], model_name, model_vs_human_multimatch, humans_multimatch, model_color)
            ax_index += 1

        # Get plot limits
        min_x, max_x = (1, 0)
        min_y, max_y = (1, 0)
        for ax in axs:
            min_x, max_x = min(min(ax.get_xlim()), min_x), max(max(ax.get_xlim()), max_x)
            min_y, max_y = min(min(ax.get_ylim()), min_y), max(max(ax.get_ylim()), max_y)
        
        for ax in axs:
            # Plot diagonal
            lims = [np.min([min_x, min_y]), np.max([max_x, max_y])]
            ax.plot(lims, lims, linestyle='dashed', c='.3')

            ax.set(xlabel='Model vs human multimatch mean', ylabel='Human multimatch mean')
            ax.label_outer()
            ax.set_box_aspect(1)

        fig.suptitle(self.dataset_name + ' dataset')
        plt.savefig(path.join(save_path, 'Multimatch against humans.png'))
        plt.show()

    def add_to_plot(self, ax, model_name, multimatch_values_per_image_x, multimatch_values_per_image_y, plot_color):
        x_vector = []
        y_vector = []
        trials_names = []
        for image_name in multimatch_values_per_image_x:
            if not image_name in multimatch_values_per_image_y:
                continue

            # Exclude temporal dimension and compute the mean of all the others dimensions: vector, direction, length, position
            value_x = np.mean(multimatch_values_per_image_x[image_name][:-1])
            value_y = np.mean(multimatch_values_per_image_y[image_name][:-1])

            trials_names.append(image_name)

            x_vector.append(value_x)
            y_vector.append(value_y)

        # Set same scale for every dataset
        ax.set_ylim(0.65, 1.0)
        ax.set_xlim(0.65, 1.0)
        # Plot multimatch
        ax.scatter(x_vector, y_vector, color=plot_color, alpha=0.5)

        # Plot linear regression
        x_linear   = np.array(x_vector)[:, np.newaxis]
        m, _, _, _ = np.linalg.lstsq(x_linear, y_vector, rcond=None)
        ax.plot(x_vector, m * x_linear, linestyle=(0, (5, 5)), color='purple', alpha=0.6)

        ax.set_title(model_name)

        # Get most similar to less similar trials names
        scores_diff = np.array(x_vector) - np.array(y_vector)
        # self.print_trials_names_in_similarity_order(scores_diff, trials_names, model_name)
    
    def save_results(self, save_path, filename):
        if self.null_object: return

        dataset_metrics_file = path.join(save_path, filename)
        dataset_metrics      = utils.load_dict_from_json(dataset_metrics_file)

        for model in self.multimatch_values:
            model_vs_human_multimatch = utils.get_random_subset(self.multimatch_values[model]['model_vs_humans'], size=self.number_of_images)
            humans_multimatch         = utils.get_random_subset(self.multimatch_values[model]['human_mean'], size=self.number_of_images)

            corr_coef_pvalue, mm_avg, model_vs_humans_mm, hmm_avg, humans_mm = self.process_results(model_vs_human_multimatch, humans_multimatch)
            metrics  = {'Corr': corr_coef_pvalue[0], 'AvgMM': mm_avg, 'MMvec': model_vs_humans_mm[0], 'MMdir': model_vs_humans_mm[1], \
                'MMlen': model_vs_humans_mm[2], 'MMpos': model_vs_humans_mm[3]}
            hmetrics = {'AvgMM': hmm_avg, 'MMvec': humans_mm[0], 'MMdir': humans_mm[1], 'MMlen': humans_mm[2], 'MMpos': humans_mm[3]}

            utils.update_dict(dataset_metrics, model, metrics)
            
            # Subjects' score is computed in grid size
            utils.update_dict(dataset_metrics, 'Humans', hmetrics)
        
        utils.save_to_json(dataset_metrics_file, dataset_metrics)
    
    def process_results(self, model_vs_human_multimatch, humans_multimatch):
        model_vs_humans_mm   = [[], [], [], []]
        model_vs_humans_mean = []
        humans_mm   = [[], [], [], []]
        humans_mean = []
        for image_name in model_vs_human_multimatch:
            if not image_name in humans_multimatch:
                continue

            # Exclude temporal dimension and keep all the others dimensions: vector, direction, length, position
            img_model_vs_human = model_vs_human_multimatch[image_name][:-1]
            img_humans         = humans_multimatch[image_name][:-1]

            for index, dim in enumerate(model_vs_humans_mm):
                dim.append(img_model_vs_human[index])
            for index, dim in enumerate(humans_mm):
                dim.append(img_humans[index])
            model_vs_humans_mean.append(np.mean(img_model_vs_human))
            humans_mean.append(np.mean(img_humans))

        corr_coef_pvalue = pearsonr(model_vs_humans_mean, humans_mean)
        
        mm_avg    = np.round(np.mean(model_vs_humans_mean), 3)
        hmm_avg   = np.round(np.mean(humans_mean), 3)

        for index, dim in enumerate(model_vs_humans_mm):
            model_vs_humans_mm[index] = [np.round(np.mean(dim), 3), np.round(np.std(dim), 3)]

        for index, dim in enumerate(humans_mm):
            humans_mm[index] = [np.round(np.mean(dim), 3), np.round(np.std(dim), 3)]

        return corr_coef_pvalue, mm_avg, model_vs_humans_mm, hmm_avg, humans_mm

    def print_trials_names_in_similarity_order(self, scores_diff, trials_names, model_name):
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
        print('Dataset: ' + self.dataset_name + '. Model: ' + model_name + '. Most similar to least similar trials')
        print('Left half:')
        print(scores_left_half[:10])
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def add_model_vs_humans_mean_per_image(self, model_name, model_scanpaths, model_color):
        " For each scanpath produced by the model, multimatch is calculated against the scanpath of that same image for every human subject "
        " The mean is computed for each image "
        " Output is a dictionary where the image names are the keys and the multimatch means are the values "

        if self.null_object:
            return

        multimatch_model_vs_humans_mean_per_image = {}
        total_values_per_image   = {}
        subjects_scanpaths_files = listdir(self.human_scanpaths_dir)
        print('[Multi-Match] Computing human-model values for ' + model_name + ' in ' + self.dataset_name + ' dataset')
        for subject_filename in subjects_scanpaths_files:
            subject_scanpaths = utils.load_dict_from_json(path.join(self.human_scanpaths_dir, subject_filename))
            for image_name in model_scanpaths:
                if not(image_name in subject_scanpaths):
                    continue

                model_trial_info   = model_scanpaths[image_name]
                subject_trial_info = subject_scanpaths[image_name]

                receptive_size = utils.get_dims(model_trial_info, subject_trial_info, key='receptive')
                screen_size    = utils.get_dims(model_trial_info, subject_trial_info, key='image')

                self.add_multimatch_to_dict(image_name, subject_trial_info, model_trial_info, multimatch_model_vs_humans_mean_per_image,
                    total_values_per_image, screen_size, receptive_size)

        # Compute mean per image
        for image_name in multimatch_model_vs_humans_mean_per_image:
            multimatch_model_vs_humans_mean_per_image[image_name] = (np.divide(multimatch_model_vs_humans_mean_per_image[image_name], total_values_per_image[image_name])).tolist()

        self.multimatch_values[model_name]['model_vs_humans'] = multimatch_model_vs_humans_mean_per_image
        self.multimatch_values[model_name]['plot_color'] = model_color

    def load_human_mean_per_image(self, model_name, model_scanpaths):
        " For each human subject, multimatch is computed against every other human subject, for each trial "
        " To be consistent with the model's scanpaths, human scanpaths are rescaled to match the model's size "
        " The mean is computed for each trial (i.e. for each image) "
        " Output is a dictionary where the image names are the keys and the multimatch means are the values "

        if self.null_object:
            return

        multimatch_human_mean_per_image = {}
        # Check if it was already computed
        multimatch_human_mean_json_file = path.join(path.join(self.dataset_results_dir, model_name), 'multimatch_human_mean_per_image.json')
        if path.exists(multimatch_human_mean_json_file):
            print('[Multi-Match] Loaded previously computed within human values for ' + model_name + ' in ' + self.dataset_name + ' dataset')
            multimatch_human_mean_per_image = utils.load_dict_from_json(multimatch_human_mean_json_file)
        else:
            print('[Multi-Match] Computing within human values for ' + model_name + ' in ' + self.dataset_name + ' dataset')
            total_values_per_image = {}
            # Compute multimatch for each image for every pair of subjects
            subjects_scanpaths_files = listdir(self.human_scanpaths_dir)
            for subject_filename in list(subjects_scanpaths_files):
                subjects_scanpaths_files.remove(subject_filename) 
                subject_scanpaths = utils.load_dict_from_json(path.join(self.human_scanpaths_dir, subject_filename))
                for subject_to_compare_filename in subjects_scanpaths_files:
                    subject_to_compare_scanpaths = utils.load_dict_from_json(path.join(self.human_scanpaths_dir, subject_to_compare_filename))
                    for image_name in subject_scanpaths:
                        if not (image_name in subject_to_compare_scanpaths and image_name in model_scanpaths):
                            continue

                        subject_trial_info = subject_scanpaths[image_name]
                        subject_to_compare_trial_info = subject_to_compare_scanpaths[image_name]

                        receptive_size = utils.get_dims(model_scanpaths[image_name], subject_trial_info, key='receptive')
                        screen_size    = utils.get_dims(model_scanpaths[image_name], subject_trial_info, key='image')

                        self.add_multimatch_to_dict(image_name, subject_trial_info, subject_to_compare_trial_info, multimatch_human_mean_per_image,
                            total_values_per_image, screen_size, receptive_size)

            # Compute mean per image
            for image_name in multimatch_human_mean_per_image:
                multimatch_human_mean_per_image[image_name] = (np.divide(multimatch_human_mean_per_image[image_name], total_values_per_image[image_name])).tolist()
            
            utils.save_to_json(multimatch_human_mean_json_file, multimatch_human_mean_per_image)

        self.multimatch_values[model_name] = {'human_mean' : multimatch_human_mean_per_image}

    def add_multimatch_to_dict(self, image_name, trial_info, trial_to_compare_info, multimatch_dict, counter_dict, screen_size, receptive_size):
        multimatch_trial_values = self.compute_multimatch(trial_info, trial_to_compare_info, screen_size, receptive_size)

        if multimatch_trial_values:
            if image_name in multimatch_dict:
                multimatch_values_acum = multimatch_dict[image_name]
                multimatch_dict[image_name] = np.add(multimatch_values_acum, multimatch_trial_values)
                counter_dict[image_name] += 1 
            else:
                multimatch_dict[image_name] = multimatch_trial_values
                counter_dict[image_name] = 1

    def compute_multimatch(self, trial_info, trial_to_compare_info, screen_size, receptive_size):
        target_found = trial_info['target_found'] and trial_to_compare_info['target_found']
        if not target_found:
           return []

        trial_scanpath_X, trial_scanpath_Y = trial_info['X'], trial_info['Y']
        trial_to_compare_scanpath_X, trial_to_compare_scanpath_Y = trial_to_compare_info['X'], trial_to_compare_info['Y']

        # Rescale accordingly
        trial_scanpath_X, trial_scanpath_Y = utils.rescale_and_crop(trial_info, screen_size, receptive_size)
        trial_to_compare_scanpath_X, trial_to_compare_scanpath_Y = utils.rescale_and_crop(trial_to_compare_info, screen_size, receptive_size)

        trial_scanpath_length            = len(trial_scanpath_X)
        trial_to_compare_scanpath_length = len(trial_to_compare_scanpath_X)

        trial_scanpath_time            = utils.get_scanpath_time(trial_info, trial_scanpath_length)
        trial_to_compare_scanpath_time = utils.get_scanpath_time(trial_to_compare_info, trial_to_compare_scanpath_length)

        # Multimatch can't be computed for scanpaths with length shorter than 3
        if trial_scanpath_length < 3 or trial_to_compare_scanpath_length < 3:
            return []

        trial_scanpath = np.array(list(zip(trial_scanpath_X, trial_scanpath_Y, trial_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
        trial_to_compare_scanpath = np.array(list(zip(trial_to_compare_scanpath_X, trial_to_compare_scanpath_Y, trial_to_compare_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])

        return mm.docomparison(trial_scanpath, trial_to_compare_scanpath, (screen_size[1], screen_size[0]))