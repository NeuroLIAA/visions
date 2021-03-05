import json
import matplotlib.pyplot as plt
from os import listdir

results_dir = '../Results/'
max_scanpath_length = 30

def main():
    datasets_results_dirs = listdir(results_dir)
    for dataset_name in datasets_results_dirs:
        models_results_dirs = listdir(results_dir + dataset_name)
        fig, ax = plt.subplots()
        for model_name in models_results_dirs:
            model_scanpaths_file = results_dir + dataset_name + '/' + model_name + '/Scanpaths.json'
            with open(model_scanpaths_file, 'r') as fp:
                model_scanpaths = json.load(fp)
            fixations_until_target_found = []
            for index in range(0, max_scanpath_length):
                fixations_until_target_found.append(0)
            for image_name in model_scanpaths.keys():
                scanpath_info = model_scanpaths[image_name]
                scanpath_length = len(scanpath_info['X'])

                if (scanpath_length < max_scanpath_length + 1) and scanpath_info['target_found']:
                    for index in range(scanpath_length - 1, max_scanpath_length):
                        fixations_until_target_found[index] += 1
                
                cumulative_performance = list(map(lambda x: float(x) / len(model_scanpaths.keys()), fixations_until_target_found))
            
            ax.plot(range(1, max_scanpath_length + 1), cumulative_performance, label = model_name)
        ax.legend()  
        dataset_name = dataset_name.replace('_', ' ') 
        plt.title(dataset_name)
        plt.xlabel('Number of fixations')
        plt.ylabel('Cumulative performance')
        plt.show()

main()
