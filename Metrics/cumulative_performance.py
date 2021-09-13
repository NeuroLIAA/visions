import matplotlib.pyplot as plt
import json
import numpy as np
from os import listdir

class Cumulative_performance:
    def __init__(self, dataset_name, number_of_images, max_scanpath_length):
        self.dataset_name     = dataset_name
        self.number_of_images = number_of_images
        self.max_scanpath_length = max_scanpath_length

        self.subjects_cumulative_performance = []

    def add_model(self, model_name, model_scanpaths):
        # If a model has performed visual search on a small subset of images (less than 80%), it is not included in the metric
        too_few_images = len(model_scanpaths.keys()) < self.number_of_images * 0.8)
        if not too_few_images:
            model_cumulative_performance = self.compute_cumulative_performance(model_scanpaths)
            self.subjects_cumulative_performance.append({'subject': model_name, 'cumulative_performance': model_cumulative_performance})
    
    def add_human_mean(self, humans_scanpaths_dir):
        humans_cumulative_performance = []
        humans_scanpaths_files = listdir(humans_scanpaths_dir)
        for human_scanpaths_file in humans_scanpaths_files:
            with open(path.join(humans_scanpaths_dir, human_scanpaths_file), 'r') as fp:
                human_scanpaths = json.load(fp)
            if self.dataset_name == 'cIBS':
                humans_cumulative_performance.append(self.compute_human_cumulative_performance_cIBS(human_scanpaths))
            else:
                humans_cumulative_performance.append(self.compute_cumulative_performance(human_scanpaths))
        
        # cIBS dataset caps the number of maximum saccades for humans at 2, 4, 8 and 12
        # Therefore, cumulative performance is calculated at fixation number 3, 5, 9 and 13
        if self.dataset_name == 'cIBS':
            number_of_subjects = len(humans_scanpaths_files)
            humans_cumulative_performance_mean = [np.empty(n) for n in np.repeat(number_of_subjects, 4)]
            subject_index = 0
            for subject_cumulative_performance in humans_cumulative_performance:
                humans_cumulative_performance_mean[0][subject_index] = subject_cumulative_performance[3]
                humans_cumulative_performance_mean[1][subject_index] = subject_cumulative_performance[5]
                humans_cumulative_performance_mean[2][subject_index] = subject_cumulative_performance[9]
                humans_cumulative_performance_mean[3][subject_index] = subject_cumulative_performance[13]
                subject_index += 1
        else:
            humans_cumulative_performance_mean = np.mean(np.array(humans_cumulative_performance), axis=0)
        self.subjects_cumulative_performance.append({'subject': 'Humans', 'cumulative_performance': humans_cumulative_performance_mean})

    def compute_cumulative_performance(self, scanpaths):
        """ At index i, this array holds the number of targets found in i or less fixations """
        targets_found_at_fixation_number = []
        for index in range(self.max_scanpath_length + 1):
            targets_found_at_fixation_number.append(0)

        for image_name in scanpaths.keys():
            scanpath_info   = scanpaths[image_name]
            scanpath_length = len(scanpath_info['X'])

            if scanpath_length <= self.max_scanpath_length and scanpath_info['target_found']:
                for index in range(scanpath_length, self.max_scanpath_length + 1):
                    targets_found_at_fixation_number[index] += 1
            
        subject_cumulative_performance = list(map(lambda x: float(x) / self.number_of_images, targets_found_at_fixation_number))
        
        return subject_cumulative_performance

    def compute_human_cumulative_performance_cIBS(self, scanpaths):
        """ Since the cIBS dataset limits the number of saccades a subject can do at a certain trial, 
            human cumulative performance is measured by counting the number of succesful trials in each specific limit (3, 5, 9 or 13 fixations)
            divided by the total number of trials with that limit. """
        cumulative_performance_at_particular_fixations = []
        for index in range(self.max_scanpath_length + 1):
            cumulative_performance_at_particular_fixations.append(0)

        number_of_scanpaths_with_three_fixations_max = 0
        number_of_scanpaths_with_five_fixations_max = 0
        number_of_scanpaths_with_nine_fixations_max = 0
        number_of_scanpaths_with_thirteen_fixations_max = 0

        for image_name in scanpaths.keys():
            scanpath_info   = scanpaths[image_name]
            scanpath_length = len(scanpath_info['X'])
            scanpath_max_fixations = scanpath_info['max_fixations']

            if scanpath_max_fixations == 3:
                number_of_scanpaths_with_three_fixations_max += 1
            elif scanpath_max_fixations == 5:
                number_of_scanpaths_with_five_fixations_max += 1
            elif scanpath_max_fixations == 9:
                number_of_scanpaths_with_nine_fixations_max += 1
            else:
                number_of_scanpaths_with_thirteen_fixations_max += 1

            if (scanpath_length <= scanpath_max_fixations) and (scanpath_max_fixations <= self.max_scanpath_length) and scanpath_info['target_found']:
                cumulative_performance_at_particular_fixations[scanpath_max_fixations] += 1
            
        cumulative_performance_at_particular_fixations[3]  = cumulative_performance_at_particular_fixations[3] / number_of_scanpaths_with_three_fixations_max
        cumulative_performance_at_particular_fixations[5]  = cumulative_performance_at_particular_fixations[5] / number_of_scanpaths_with_five_fixations_max
        cumulative_performance_at_particular_fixations[9]  = cumulative_performance_at_particular_fixations[9] / number_of_scanpaths_with_nine_fixations_max
        cumulative_performance_at_particular_fixations[13] = cumulative_performance_at_particular_fixations[13] / number_of_scanpaths_with_thirteen_fixations_max

        return cumulative_performance_at_particular_fixations

    def plot(self, save_path):
        fig, ax = plt.subplots()
        for subject in self.subjects_cumulative_performance:
            subject_name = subject['subject']
            subject_cumulative_performance = subject['cumulative_performance'] 

            if subject_name == 'Humans' and self.dataset_name == 'cIBS':
                ax.boxplot(subject_cumulative_performance, notch=True, vert=True, whiskerprops={'linestyle': (0, (5, 10))}, \
                    flierprops={'marker': '+', 'markeredgecolor': 'red'}, positions=[3, 5, 9, 13])
            else:
                ax.plot(range(1, self.max_scanpath_length + 1), subject_cumulative_performance[1:], label = subject_name)

        ax.legend()  
        dataset_name = self.dataset_name + ' dataset'

        plt.title(dataset_name)
        plt.yticks(np.arange(0, 1, 0.1))
        plt.xlabel('Number of fixations')
        plt.ylabel('Cumulative performance')
        plt.savefig(path.join(save_path, 'Cumulative performance.png'))
        plt.show()