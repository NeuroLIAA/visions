import matplotlib.pyplot as plt
import json
import numpy as np
from os import listdir

class Cumulative_performance:
    def __init__(self, dataset_name, max_scanpath_length):
        self.subjects_cumulative_performance = []
        self.max_scanpath_length = max_scanpath_length
        self.dataset_name = dataset_name

    def add_model(self, model_name, scanpaths):
        model_cumulative_performance = self.compute_cumulative_performance(scanpaths)
        self.subjects_cumulative_performance.append({'subject': model_name, 'cumulative_performance': model_cumulative_performance})
    
    def add_human_average(self, humans_scanpaths_dir):
        humans_cumulative_performance = []
        humans_scanpaths_files = listdir(humans_scanpaths_dir)
        for human_scanpaths_file in humans_scanpaths_files:
            with open(humans_scanpaths_dir + human_scanpaths_file, 'r') as fp:
                human_scanpaths = json.load(fp)
            humans_cumulative_performance.append(self.compute_cumulative_performance(human_scanpaths))
        
        humans_cumulative_performance_average = np.mean(np.array(humans_cumulative_performance), axis=0)
        self.subjects_cumulative_performance.append({'subject': 'Humans', 'cumulative_performance': humans_cumulative_performance_average})

    def compute_cumulative_performance(self, scanpaths):
        # At index i, this array holds the number of targets found in i or less fixations
        targets_found_at_fixation_number = []
        for index in range(0, self.max_scanpath_length):
            targets_found_at_fixation_number.append(0)

        for image_name in scanpaths.keys():
            scanpath_info   = scanpaths[image_name]
            scanpath_length = len(scanpath_info['X'])

            if (scanpath_length < self.max_scanpath_length + 1) and scanpath_info['target_found']:
                for index in range(scanpath_length - 1, self.max_scanpath_length):
                    targets_found_at_fixation_number[index] += 1
            
            subject_cumulative_performance = list(map(lambda x: float(x) / len(scanpaths.keys()), targets_found_at_fixation_number))
        
        return subject_cumulative_performance

    def plot(self):
        fig, ax = plt.subplots()
        for subject in self.subjects_cumulative_performance:
            subject_name = subject['subject']
            subject_cumulative_performance = subject['cumulative_performance'] 
        
            ax.plot(range(1, self.max_scanpath_length + 1), subject_cumulative_performance, label = subject_name)

        ax.legend()  
        dataset_name = self.dataset_name + ' dataset'
        plt.title(dataset_name)
        plt.xlabel('Number of fixations')
        plt.ylabel('Cumulative performance')
        plt.show()