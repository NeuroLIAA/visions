import IVSN
import image_preprocessing
import compute_scanpaths

stimuli_dir = 'stimuli/'
target_dir = 'target/'
chopped_dir = 'choppednaturaldesign/'
save_path = 'results/Scanpaths.json'
stimuli_size = (1028, 1280)
max_fixations  = 80
receptive_size = 200

def main():
    print('Preprocessing images...')
    image_preprocessing.chop_stimuli(stimuli_dir, chopped_dir, stimuli_size)
    print('Running model...')
    IVSN.run(stimuli_dir, target_dir, chopped_dir)
    print('Computing scanpaths...')
    compute_scanpaths.parse_model_data(stimuli_dir, chopped_dir, stimuli_size, max_fixations, receptive_size, save_path)

main()
