import json
from os import makedirs, listdir, path, remove, cpu_count

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

def load_config(config_dir, config_name, image_size, number_of_processes, save_probability_maps, human_scanpaths, checkpoint):
    if checkpoint:
        config = checkpoint['configuration']
    else:
        config = load_dict_from_json(config_dir + config_name + '.json')

    config['image_size'] = image_size

    if number_of_processes == 'all':
        config['proc_number'] = cpu_count()
    else:
        config['proc_number'] = int(number_of_processes)

    config['save_probability_maps'] = save_probability_maps or bool(human_scanpaths)
    config['human_scanpaths']       = human_scanpaths

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
    if config['human_scanpaths']:
        print('Human subject\'s scanpaths will be used as model\'s fixations')
    if config['save_probability_maps']:
        print('Probability maps will be saved for each saccade')
    if config['save_similarity_maps']:
        print('Target similarity maps will be saved for each image')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    
    return config

def load_dataset_info(dataset_info_file):
    return load_dict_from_json(dataset_info_file)

def load_human_scanpaths(human_scanpaths_dir, human_subject):
    if human_subject is None:
        return {}

    human_scanpaths_files = listdir(human_scanpaths_dir)
    human_subject_str     = str(human_subject)
    if human_subject < 10: human_subject_str = '0' + human_subject_str
    human_subject_file    = 'subj' + human_subject_str + '_scanpaths.json'
    if not human_subject_file in human_scanpaths_files:
        raise NameError('Scanpaths for human subject ' + human_subject_str + ' not found!')
    
    human_scanpaths = load_dict_from_json(path.join(human_scanpaths_dir, human_subject_file))

    return human_scanpaths

def load_trials_properties(trials_properties_file, image_name, image_range, human_scanpaths, checkpoint):
    trials_properties = load_dict_from_json(trials_properties_file)
    
    if checkpoint:
        trials_properties = checkpoint['trials_properties']
    elif image_name is not None:
        trials_properties = get_trial_properties_for_image(trials_properties, image_name)
    elif image_range is not None:
        trials_properties = get_trial_properties_in_range(trials_properties, image_range)
    
    if human_scanpaths:
        trials_properties = get_trial_properties_for_subject(trials_properties, human_scanpaths)

    return trials_properties

def get_trial_properties_for_subject(trials_properties, human_scanpaths):
    trial_properties = []
    for trial in trials_properties:
        if trial['image'] in human_scanpaths:
            trial_properties.append(trial)
    
    if not trial_properties:
        raise ValueError('Human subject does not have any scanpaths for the images specified')

    return trial_properties

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

def create_output_folders(save_path, config_name, image_name, image_range, human_subject):
    output_path = save_path

    if image_name is not None:
        output_path = path.join(output_path, image_name[:-4] + '/')
    if image_range is not None:
        output_path = path.join(output_path, 'range_' + str(image_range[0]) + '-' + str(image_range[1]) + '/')
    if human_subject is not None:
        human_subject_str = str(human_subject)
        if human_subject < 10: human_subject_str = '0' + human_subject_str
        output_path = path.join(output_path, 'human_subject_' + human_subject_str + '/')

    makedirs(output_path, exist_ok=True)

    return output_path

def load_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)