from . import utils
import time
from tqdm import tqdm
from .model import VisualSearchModel as VisualSearchModel
from Metrics.scripts import human_scanpath_prediction

def start(trials_properties, exp_info, imgs_path, tgs_path, human_scanpaths, cfg_file, vgg16_weights, dataset_name, output_path):
    ecc_param = utils.load_dict_from_json(str(cfg_file))
    model_cfg = utils.build_modelcfg(ecc_param, str(vgg16_weights))

    vs_model = VisualSearchModel(model_cfg)
    vs_model.load_exp_info(exp_info, corner_bias=exp_info['corner_bias'])

    scanpaths, targets_found = {}, 0
    t0 = time.time()
    for trial in tqdm(trials_properties):
        img_path = imgs_path / trial['image']
        tg_path  = tgs_path / trial['target']
        img_size = (trial['image_height'], trial['image_width'])
        tg_bbox  = [trial['target_matched_row'], trial['target_matched_column'], \
            trial['target_height'] + trial['target_matched_row'], trial['target_width'] + trial['target_matched_column']]
        tg_bbox  = [utils.rescale_coordinate(tg_bbox[i], img_size[i % 2 == 1], exp_info['stim_shape'][i % 2 == 1]) for i in range(len(tg_bbox))]
        initial_fix = (trial['initial_fixation_row'], trial['initial_fixation_column'])
        initial_fix = [utils.rescale_coordinate(initial_fix[i], img_size[i], exp_info['stim_shape'][i]) for i in range(len(initial_fix))]

        human_scanpath = utils.get_scanpath(human_scanpaths, trial['image'])

        trial_fixations, target_found = vs_model.start_search(img_path, tg_path, tg_bbox, initial_fix, human_scanpath, trial['image'], output_path)
        targets_found += target_found
        
        if human_scanpath:
            human_scanpath_prediction.save_scanpath_prediction_metrics(human_scanpath, trial['image'], output_path)

        scanpaths[trial['image']] = utils.build_trialscanpath(trial_fixations, target_found, tg_bbox, img_size=exp_info['stim_shape'], 
            max_fix=exp_info['NumFix'], receptive_size=exp_info['ior_size'], tg_object=trial['target_object'], dataset=dataset_name)
        
    print('Total targets found: {}/{}'.format(targets_found, len(trials_properties)))
    print('Total time: {:.2f} seconds'.format(time.time() - t0))
    
    if human_scanpaths:
        utils.save_scanpaths(human_scanpaths, output_path, filename='Subject_scanpaths.json')
    else:
        utils.save_scanpaths(scanpaths, output_path)