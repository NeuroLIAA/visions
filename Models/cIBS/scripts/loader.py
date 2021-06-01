import json
from os import makedirs, path, remove, cpu_count

def load_checkpoint(output_path):
    checkpoint = {}
    checkpoint_file = output_path + 'checkpoint.json'
    if path.exists(checkpoint_file):
        answer = input('Checkpoint found! Resume execution? (Y/N): ').upper()
        if answer not in ['Y', 'N']:
            print('Invalid answer. Please try again')
            load_checkpoint(output_path)
        if answer == 'Y':
            checkpoint = load_dict_from_json(checkpoint_file)
            print('Checkpoint loaded. Resuming execution...\n')
        if answer == 'N':
            remove(checkpoint_file)
            print('Checkpoint deleted\n')
    
    return checkpoint

def load_config(config_dir, config_name, number_of_processes, checkpoint):
    if checkpoint:
        config = checkpoint['configuration']
    else:
        config = load_dict_from_json(config_dir + config_name + '.json')

    if number_of_processes == 'all':
        config['proc_number'] = cpu_count()
    else:
        config['proc_number'] = int(number_of_processes)

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    if checkpoint:
        print('Successfully loaded previously used configuration')
    else:
        print('Successfully loaded ' + config_name + '.json!')
    print('Search model: ' + config['search_model'])
    print('Target similarity: ' + config['target_similarity'])
    print('Prior: ' + config['prior'])
    print('Max. saccades: ' + str(config['max_saccades']))
    print('Cell size: ' + str(config['cell_size']))
    print('Scale factor: ' + str(config['scale_factor']))
    print('Additive shift: ' + str(config['additive_shift']))
    print('Random seed: ' + str(config['seed']))
    if config['proc_number'] > 1:
        print('Multiprocessing is ENABLED!')
    else:
        print('Multiprocessing is DISABLED')
    if config['save_probability_maps']:
        print('Probability maps will be saved for each saccade')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    return config

def load_dataset_info(dataset_info_file):
    return load_dict_from_json(dataset_info_file)

def load_trials_properties(trials_properties_file, image_name, image_range, checkpoint):
    trials_properties = load_dict_from_json(trials_properties_file)
    
    if checkpoint:
        trials_properties = checkpoint['trials_properties']
    elif image_name is not None:
        trials_properties = get_trial_properties_for_image(trials_properties, image_name)
    elif image_range is not None:
        trials_properties = get_trial_properties_in_range(trials_properties, image_range)

    return trials_properties

def get_trial_properties_for_image(trials_properties, image_name):
    trial_properties = []
    for trial in trials_properties:
        if trial['image'] == image_name:
            trial_properties.append(trial)
            break        
    
    if not trial_properties:
        raise NameError('Image name must be in the dataset')

    return trial_properties

def get_trial_properties_in_range(trials_properties, image_range):
    trial_properties = []
    
    counter   = 1
    min_range = image_range[0]
    max_range = image_range[1]
    for trial in trials_properties:
        if counter >= min_range and counter <= max_range:
            trial_properties.append(trial)
        
        counter += 1
    
    if not trial_properties:
        raise ValueError('Range outside of dataset\'s scope')

    return trial_properties

def create_output_folders(save_path, config_name, image_name, image_range):
    output_path = save_path + '/'

    if image_name is not None:
        output_path += image_name[:-4] + '/'
    if image_range is not None:
        output_path += 'range_' + str(image_range[0]) + '-' + str(image_range[1]) + '/'

    makedirs(output_path, exist_ok=True)

    return output_path

def load_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)