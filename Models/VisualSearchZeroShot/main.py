import IVSN
import image_preprocessing
import compute_scanpaths
import json

stimuli_dir = 'stimuli/'
target_dir = 'target/'
chopped_dir = 'choppednaturaldesign/'
save_path = 'results/Scanpaths.json'
stimuli_size = (1028, 1280)
max_fixations  = 80
receptive_size = 200

def main():
    targetsLocationsFile = open('targets_locations.json')
    targetsLocations = json.load(targetsLocationsFile)
    targetsLocationsFile.close()
    print('Preprocessing images...')
    image_preprocessing.chop_stimuli(stimuli_dir, chopped_dir, stimuli_size,targetsLocations)
    print('Running model...')
    IVSN.run(stimuli_dir, target_dir, chopped_dir, targetsLocations)
    print('Computing scanpaths...')
    compute_scanpaths.parse_model_data(stimuli_dir, chopped_dir, stimuli_size, max_fixations, receptive_size, save_path, targetsLocations)

main()
