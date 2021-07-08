import IVSN
import image_preprocessing
import compute_scanpaths
import json

# dataset_name = 'IVSN Natural Design Dataset'
# stimuli_dir = '../../Datasets/IVSN/stimuli/'
# target_dir = '../../Datasets/IVSN/target/'
# trials_properties_file = '../../Datasets/IVSN/trials_properties.json'
# save_path = '../../Results/IVSN_dataset/IVSN/'
# stimuli_size = (1024, 1280)
# max_fixations  = 80
# receptive_size = 200

dataset_name = 'cIBS Dataset'
stimuli_dir = '../../Datasets/cIBS/images/'
target_dir = '../../Datasets/cIBS/templates/'
trials_properties_file = '../../Datasets/cIBS/trials_properties.json'
save_path = '../../Results/cIBS_dataset/IVSN/'
stimuli_size = (768, 1024)
max_fixations  = 16
receptive_size = 72

# dataset_name = 'COCOSearch18 Dataset'
# stimuli_dir = '../../Datasets/COCOSearch18/images/'
# target_dir = '../../Datasets/COCOSearch18/templates/'
# trials_properties_file = '../../Datasets/COCOSearch18/trials_properties.json'
# save_path = '../../Results/COCOSearch18_dataset/IVSN/'
# stimuli_size = (1050, 1680)
# max_fixations  = 10
# receptive_size = 54

def main():
    with open(trials_properties_file) as fp:
        trials_properties = json.load(fp)
    
    chopped_dir = 'chopped_images/' + dataset_name + '/'

    print('Preprocessing images...')
    image_preprocessing.chop_stimuli(stimuli_dir, chopped_dir, stimuli_size, trials_properties)
    print('Running model...')
    IVSN.run(stimuli_dir, target_dir, chopped_dir, trials_properties)
    print('Computing scanpaths...')
    compute_scanpaths.parse_model_data(stimuli_dir, chopped_dir, stimuli_size, max_fixations, receptive_size, save_path, trials_properties, dataset_name)

main()
