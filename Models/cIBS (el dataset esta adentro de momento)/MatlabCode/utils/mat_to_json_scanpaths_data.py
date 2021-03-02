from scipy.io import loadmat
import json
import os

cfg_dir = '../out_models/deepgaze/correlation/a_3_b_4_tam_celda_32/cfg/'
scanpaths_dir= '../out_models/deepgaze/correlation/a_3_b_4_tam_celda_32/scanpath/'
cfg_files = os.listdir(cfg_dir)
scanpaths_files = os.listdir(scanpaths_dir)
save_path = '../../../../Results/cIBS/cIBS_dataset/'

window_size = (32, 32)


scanpath_data = dict()
for cfg_file in cfg_files:
    if cfg_file == 'time.mat':
        continue
    cfg_info = loadmat(cfg_dir + cfg_file)
    cfg_split_file_name = cfg_file.split('_')
    cfg_id = cfg_split_file_name[len(cfg_split_file_name) - 1][:-4]
    
    scanpath_info = loadmat(scanpaths_dir + 'scanpath_' + cfg_id + '.mat')
    cfg_info = cfg_info['cfg']
    scanpath_info = scanpath_info['scanpath']
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Processing ' + cfg_file)
    print('\n')


    image_name   = cfg_info['imgname'][0][0][0]
    image_height = int(cfg_info['image_size'][0][0][0][0])
    image_width  = int(cfg_info['image_size'][0][0][0][1])


    #target_bbox = no entendi el formato que usa
    #target_found = alta paja

    max_fixations = int(cfg_info['nsaccades_thr'][0][0][0][0]) #si cuentan las sacadas, las fijaciones no serían cantidad de sacadas + 1?
    # Subtract one, since Python indexes images from zero
    #fix_posX = conversion rancia
    #fix_posY = conversion rancia


    #if (len(fix_posX) == 0):
        #print("F en el chat")
        #continue

    #number_of_fixations = len(fix_posX)
    #last_fixation_X = fix_posX[number_of_fixations - 1]
    #last_fixation_Y = fix_posY[number_of_fixations - 1]
    #between_bounds = (target_bbox[0] - window_size[0] <= last_fixation_Y) and (target_bbox[2] + window_size[0] >= last_fixation_Y) and (target_bbox[1] - window_size[1] <= last_fixation_X) and (target_bbox[3] + window_size[1] >= last_fixation_X)
    #if (target_found):
     #   if not(between_bounds):
       #     target_found = False

        scanpath_data[image_name] = {"dataset" : "cIBS Dataset", "subject" : "cIBS model", "image_height" : image_height, "image_width" : image_width, "target_object" : "TBD", "max_fixations" : max_fixations}
     #"target_found" : target_found, "X" : fix_posX.tolist(), "Y" : fix_posY.tolist(), 
    
if not(os.path.exists(save_path)):
    os.mkdir(save_path)

with open(save_path + 'Scanpaths.json', 'w') as fp:
    json.dump(scanpath_data, fp, indent = 4)


