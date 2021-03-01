import IVSN
import image_preprocessing
import compute_scanpaths
import json

chopped_dir = 'choppednaturaldesign/'

stimuli_dir = '../../Datasets/IVSN/stimuli/'
target_dir = '../../Datasets/IVSN/target/'
targets_locations_file = '../../Datasets/IVSN/targets_locations.json'
save_path = '../Results/IVSN/IVSN_dataset/Scanpaths.json'
stimuli_size = (1028, 1280)
max_fixations  = 80
receptive_size = 200

# stimuli_dir = '../../Datasets/cIBS/images/'
# target_dir = '../../Datasets/cIBS/templates/'
# targets_locations_file = '../../Datasets/cIBS/targets_locations.json'
# save_path = '../Results/cIBS/cIBS_dataset/Scanpaths.json'
# stimuli_size = (768, 1024)
# max_fixations  = 80
# receptive_size = 32

def main():
    with open(targets_locations_file) as fp:
        targets_locations = json.load(fp)

    print('Preprocessing images...')
    image_preprocessing.chop_stimuli(stimuli_dir, chopped_dir, stimuli_size, targets_locations)
    print('Running model...')
    IVSN.run(stimuli_dir, target_dir, chopped_dir, targets_locations)
    print('Computing scanpaths...')
    compute_scanpaths.parse_model_data(stimuli_dir, chopped_dir, stimuli_size, max_fixations, receptive_size, save_path, targets_locations)

main()