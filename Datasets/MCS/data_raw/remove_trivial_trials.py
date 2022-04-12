import utils
from os import listdir, path

trials_properties_file = '../trials_properties.json'
dataset_info_file      = '../dataset_info.json'

trials_properties = utils.load_dict_from_json(trials_properties_file)
dataset_info      = utils.load_dict_from_json(dataset_info_file)

human_scanpaths_dir   = path.join('..', dataset_info['scanpaths_dir'])
human_scanpaths_files = listdir(human_scanpaths_dir)

receptive_size = dataset_info['receptive_size']

repeated_targets = ['c000000002591.jpg', 'c000000003080.jpg', 'c000000005809.jpg', 'c000000007504.jpg', 'c000000014412.jpg', 'c000000014855.jpg', 'c000000016898.jpg', 'c000000017892.jpg', 'c000000018952.jpg', 'c000000019941.jpg', 'c000000020145.jpg', 'c000000020517.jpg', 'c000000024019.jpg', 'c000000024582.jpg', 'c000000025195.jpg', 'c000000026109.jpg', 'c000000026247.jpg', 'c000000027151.jpg', 'c000000027840.jpg', 'c000000028385.jpg', 'c000000029586.jpg', 'c000000032328.jpg', 'c000000033413.jpg', 'c000000036796.jpg', 'c000000043601.jpg', 'c000000044792.jpg', 'c000000044810.jpg', 'c000000045419.jpg', 'c000000045768.jpg', 'c000000046142.jpg', 'c000000053842.jpg', 'c000000069200.jpg', 'c000000075325.jpg', 'c000000084651.jpg', 'c000000086875.jpg', 'c000000103513.jpg', 'c000000129485.jpg', 'c000000131595.jpg', 'c000000159754.jpg', 'c000000169766.jpg', 'c000000172784.jpg', 'c000000178248.jpg', 'c000000180588.jpg', 'c000000180623.jpg', 'c000000181938.jpg', 'c000000184148.jpg', 'c000000194753.jpg', 'c000000209092.jpg', 'c000000239350.jpg', 'c000000239613.jpg', 'c000000263727.jpg', 'c000000281191.jpg', 'c000000297415.jpg', 'c000000338118.jpg', 'c000000369509.jpg', 'c000000377649.jpg', 'c000000403521.jpg', 'c000000403553.jpg', 'c000000407091.jpg', 'c000000414244.jpg', 'c000000416387.jpg', 'c000000425473.jpg', 'c000000433607.jpg', 'c000000436780.jpg', 'c000000444181.jpg', 'c000000448533.jpg', 'c000000464634.jpg', 'c000000464839.jpg', 'c000000475485.jpg', 'c000000479941.jpg', 'c000000483786.jpg', 'c000000491182.jpg', 'c000000491510.jpg', 'c000000492580.jpg', 'c000000493333.jpg', 'c000000501648.jpg', 'c000000503497.jpg', 'c000000512797.jpg', 'c000000533004.jpg', 'c000000549478.jpg', 'c000000563725.jpg', 'c000000564733.jpg', 'm000000024023.jpg', 'm000000071099.jpg', 'm000000157744.jpg', 'm000000180794.jpg', 'm000000191806.jpg', 'm000000267189.jpg', 'm000000282855.jpg', 'm000000293340.jpg', 'm000000310140.jpg', 'm000000354804.jpg', 'm000000511165.jpg']
removed_images   = []
for trial in list(trials_properties):
    target_bbox = [trial['target_matched_row'], trial['target_matched_column'], trial['target_matched_row'] + trial['target_height'], trial['target_matched_column'] + trial['target_width']]
    initial_fixation = (trial['initial_fixation_row'], trial['initial_fixation_column'])

    if trial['image'] in repeated_targets:
        removed_images.append(trial['image'])
        trials_properties.remove(trial)

    if utils.between_bounds(target_bbox, initial_fixation[0], initial_fixation[1], receptive_size):
        removed_images.append(trial['image'])
        trials_properties.remove(trial)

trials_properties_file = '../trials_properties_cropped.json'
utils.save_to_json(trials_properties_file, trials_properties)

trivial_scanpaths = 0
target_found_removed = 0
trials_removed = 0
for subject in human_scanpaths_files:
    subject_scanpaths = utils.load_dict_from_json(path.join(human_scanpaths_dir, subject))
    for image in removed_images:
        if image in subject_scanpaths:
            scanpath = subject_scanpaths[image]
            if scanpath['target_found']:
                target_found_removed += 1
            trials_removed += 1
            del subject_scanpaths[image]
    
    for trial in dict(subject_scanpaths):
        scanpath = subject_scanpaths[trial]
        if len(scanpath['X']) == 1:
            trials_removed += 1
            trivial_scanpaths += 1
            if scanpath['target_found']: target_found_removed += 1
            del subject_scanpaths[trial]
    
    utils.save_to_json(path.join(human_scanpaths_dir, subject), subject_scanpaths)

print('Removed images: ' + str(removed_images))
print('Number of removed images: ' + str(len(removed_images)))
print('Trivial scanpaths (after removing images): ' + str(trivial_scanpaths))
print('Total trials removed: ' + str(trials_removed) + ' of which ' + str(target_found_removed) + ' had found the target')