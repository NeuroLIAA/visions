from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import numpy as np
import cv2
import math
from .ecc_net import load_eccNET, load_VGG16
from tqdm.notebook import tqdm
from skimage.filters.rank import entropy
from skimage.morphology import disk, square

def loadimg(src, target_size, rev_img_flag=0):
    """
    To load the images
    """

    img = image.load_img(src, target_size=target_size)
    x = image.img_to_array(img)
    if rev_img_flag==1:
        x = 255 - x
    x = np.expand_dims(x, axis=0)
    return x

def remove_attn(mask, x, y, ior_size=None, gt_mask=None):
    """
    To apply inhibition of return (IoR) on an object/place which has been attended by the model.
    """

    if gt_mask is None:
        mask[:, (x - int(ior_size/2)):(x + int(ior_size/2)), (y - int(ior_size/2)):(y + int(ior_size/2)), :] = 0
    else:
        for i in range(len(gt_mask)):
            gt = np.copy(gt_mask[i])
            gt = np.expand_dims(gt, axis=0)
            gt = np.expand_dims(gt, axis=3)
            if gt[0, x, y] > 0:
                mask = np.copy((1-gt)*mask)
                break
    return mask

def recog(x, y, gt, ior_size):
    """
    To test if the current attended position is the target.
    """

    fxt_xtop = x-int(ior_size/2)
    fxt_ytop = y-int(ior_size/2)
    fxt_place = gt[fxt_xtop:(fxt_xtop+ior_size), fxt_ytop:(fxt_ytop+ior_size)]

    if (np.sum(fxt_place)>0):
        return 1
    else:
        return 0

class VisualSearchModel:
    def __init__(self, model_desc, w_l=None):
        '''
        Initialization of the Visual Search Model
        '''

        self.model_desc = model_desc
        self.vgg_model_path = model_desc['vgg_model_path']
        self.w_l = w_l

        # tmp_name is basically used to tell which of the last three pooling layers were used to get the features for top-down modulation
        tmp_name = "_" + str(model_desc['out_layer'][0]) + "_" + str(model_desc['out_layer'][1]) + "_" + str(model_desc['out_layer'][2])

        # If eccentricity depth is '0' name it as IVSN -- Invariant Visual Search Network
        # Zhang, M., Feng, J., Ma, K.T. et al. Finding any Waldo with zero-shot invariant and efficient visual search. Nat Commun 9, 3730 (2018). https://doi.org/10.1038/s41467-018-06217-x
        if model_desc['ecc_depth'] == 0:
            if sum(model_desc['out_layer'][:2]) == 0:
                tmp_name = ""
            self.model_name = 'IVSN' + tmp_name
        else:
            if sum(model_desc['out_layer']) == 3:
                tmp_name = ""
            self.model_name = 'eccNET' + tmp_name

        self.model_name += model_desc["model_subname"]

    def load_exp_info(self, exp_info, corner_bias=16*4):
        """
        Load the experiment informations
        """

        self.model_desc['deg2px'] = exp_info['deg2px']
        self.gt_mask = exp_info['gt_mask']
        self.ior_size = exp_info['ior_size']
        self.NumFix = exp_info['NumFix']
        self.NumStimuli = exp_info['NumStimuli']
        self.stim_shape = exp_info['stim_shape']
        self.tar_shape = exp_info['tar_shape']
        self.eye_res = exp_info['eye_res']
        self.bg_value = exp_info['bg_value']
        self.rev_img_flag = exp_info['rev_img_flag']
        self.fix = exp_info['fix']
        self.weight_pattern = exp_info['weight_pattern']

        self.corner_bias = corner_bias
        self.model_ip_shape = (2*(self.eye_res+corner_bias), 2*(self.eye_res+corner_bias), 3)

        if self.model_desc['ecc_depth'] == 0:
            self.target_model, self.stimuli_model = load_VGG16(self.vgg_model_path, self.model_ip_shape, self.tar_shape, eccParam=self.model_desc['eccParam'], ecc_depth=self.model_desc['ecc_depth'], comp_layer=self.model_desc['comp_layer'])
        else:
            self.target_model, self.stimuli_model = load_eccNET(self.vgg_model_path, self.model_ip_shape, self.tar_shape, eccParam=self.model_desc['eccParam'], ecc_depth=self.model_desc['ecc_depth'], comp_layer=self.model_desc['comp_layer'])

        dog_size = exp_info['dog_size']
        self.win_size_l = dog_size[0][0]
        self.win_sigma_l = dog_size[0][1]

        self.win_size_u = dog_size[1][0]
        self.win_sigma_u = dog_size[1][1]

        if self.gt_mask is not None:
            self.reformat_gt_mask()

    def start_search(self, stim_path, tar_path, tg_bbox, initial_fix, debug_flag=False, attn_map_flag=False):
        """
        Perform the visual search on the given search and target image.
        """

        MMconv_l = self.__create_conv_win(tar_path)
        gt = self.__load_gt(tg_bbox)
        stimuli = self.__load_stim(stim_path)
        ip_stimuli = preprocess_input(np.uint8(stimuli))
        visual_field = self.eye_res + self.corner_bias

        # eye_res = 736; corner_bias = 64; stim_shape = [736, 896]; model_ip_shape = [1600, 1600] = [2*(eye_res+corner_bias), 2*(eye_res+corner_bias)]
        # stimuli.shape = [2336, 2496] = [stim_shape[0]+2*(eye_res+corner_bias), stim_shape[1]+2*(eye_res+corner_bias)]
        temp_stim = np.uint8(np.zeros((self.stim_shape[0] + 2*visual_field, self.stim_shape[1] + 2*visual_field)))
        if self.gt_mask is None:
            temp_stim[visual_field:self.stim_shape[0]+visual_field, visual_field:self.stim_shape[1]+visual_field] = 1
        else:
            temp_stim = np.sum(self.gt_mask, axis=0)

        stimuli_area = np.copy(temp_stim)
        mask = np.ones(stimuli.shape)

        # Add initial fixation
        saccade = []
        (x, y) = int(initial_fix[0] + visual_field), int(initial_fix[1] + visual_field)
        saccade.append((x, y))

        attn_maps = []
        vis_area_crop = []

        for k in range(self.NumFix):
            if self.gt_mask is None:
                if recog(saccade[-1][0], saccade[-1][1], gt, self.ior_size):
                    break
                mask = remove_attn(mask, saccade[-1][0], saccade[-1][1], self.ior_size, self.gt_mask)

            if self.gt_mask is not None:
                ip_stimuli = preprocess_input(np.uint8(mask*stimuli + self.bg_value*(1-mask)))

            vis_area = (ip_stimuli)[:, saccade[-1][0]-visual_field:saccade[-1][0]+visual_field, saccade[-1][1]-visual_field:saccade[-1][1]+visual_field, :]
            op_stimuli_l = self.stimuli_model.predict(vis_area)

            outf_l = []
            w_l = []
            for i in range(len(MMconv_l)):
                MMconv = MMconv_l[i]
                op_stimuli = op_stimuli_l[i]
                out = MMconv(tf.constant(op_stimuli)).numpy().reshape((op_stimuli.shape[1], op_stimuli.shape[2]))

                w_l.append(np.max(out))
                out = cv2.normalize(out, None, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                outf_l.append(out)

            w_l = np.array(w_l)
            if self.weight_pattern == 'l':
                w_l = w_l*0 + 1

            if self.w_l == None:
                pass
            else:
                w_l = np.array(self.w_l)

            w_l = np.array(self.model_desc['out_layer'])*w_l

            outf = np.zeros(outf_l[0].shape)
            w_l = w_l/np.sum(w_l)
            for i in range(w_l.shape[0]):
                outf += outf_l[i]*w_l[i]

            out = outf

            for j in range(3):
                g1 = cv2.GaussianBlur(np.copy(out),(self.win_size_l, self.win_size_l), self.win_sigma_l)
                g2 = cv2.GaussianBlur(np.copy(out),(self.win_size_u, self.win_size_u), self.win_sigma_u)
                out = out + (g1 - g2)

            out = cv2.resize(out, (self.model_ip_shape[0], self.model_ip_shape[1]), interpolation = cv2.INTER_AREA)

            temp_stim = np.uint8(np.zeros((self.model_ip_shape[0], self.model_ip_shape[1])))
            stim_mask = np.copy(mask[0,:,:,0]*stimuli_area)
            stim_mask = (stim_mask)[saccade[-1][0]-self.eye_res:saccade[-1][0]+self.eye_res, saccade[-1][1]-self.eye_res:saccade[-1][1]+self.eye_res]
            if self.corner_bias > 0:
                temp_stim[self.corner_bias:-self.corner_bias, self.corner_bias:-self.corner_bias] = np.copy(stim_mask)
                stim_mask = np.copy(temp_stim)

            out = out - np.min(out)
            out = out*(stim_mask)

            if debug_flag:
                attn_maps.append(out)
                temp_vis_area = np.copy((stimuli)[0, saccade[-1][0]-visual_field:saccade[-1][0]+visual_field, saccade[-1][1]-visual_field:saccade[-1][1]+visual_field, :])
                vis_area_crop.append(temp_vis_area)

            (x, y) = np.unravel_index(np.argmax(out), out.shape)
            fxn_x, fxn_y = saccade[-1][0]-visual_field+x, saccade[-1][1]-visual_field+y
            fxn_x, fxn_y = max(fxn_x, visual_field), max(fxn_y, visual_field)
            fxn_x, fxn_y = min(fxn_x, (stimuli.shape[1]-visual_field)), min(fxn_y, (stimuli.shape[2]-visual_field))
            saccade.append((fxn_x, fxn_y))

            if self.gt_mask is not None:
                x, y = saccade[-1][0], saccade[-1][1]
                mask = remove_attn(mask, x, y, self.ior_size, self.gt_mask)
                if recog(x, y, gt, self.ior_size):
                    break

        saccade = np.array(saccade)

        if debug_flag:
            return self.post_process_saccades(saccade), attn_maps, vis_area_crop
        else:
            return self.post_process_saccades(saccade)

    def __create_conv_win(self, tar_path):
        """
        Creates convolutional window using the target image to apply top-down modulation on the search image.
        """

        target = loadimg(tar_path, target_size=(self.tar_shape[0], self.tar_shape[1]), rev_img_flag=self.rev_img_flag)
        target = preprocess_input(target)
        op_target_l = self.target_model.predict(target)
        op_stimuli_l = self.stimuli_model.predict(np.zeros((1, self.model_ip_shape[0], self.model_ip_shape[1], self.model_ip_shape[2])))

        MMconv_l = []

        for i in range(len(op_stimuli_l)):
            op_target = op_target_l[i]
            op_stimuli = op_stimuli_l[i]
            MMconv = tf.keras.layers.Conv2D(1, kernel_size=(op_target.shape[1], op_target.shape[2]),
                                            input_shape=(op_stimuli.shape[1], op_stimuli.shape[2], op_stimuli.shape[3]),
                                            padding='same',
                                            use_bias=False)

            init = MMconv(tf.constant(op_stimuli))
            layer_weight = []
            layer_weight.append(op_target.reshape(MMconv.get_weights()[0].shape))
            MMconv.set_weights(layer_weight)
            MMconv_l.append(MMconv)

        return MMconv_l

    def __load_gt(self, tg_bbox):
        """
        gt is basically helper array to perform the perfect object recognition. i.e. it gives the bounding box of the region where the target is.
        FT: changed to build the gt matrix on runtime
        """
        visual_field = self.eye_res + self.corner_bias
        gt = np.zeros((self.stim_shape[0]+2*visual_field, self.stim_shape[1]+2*visual_field), dtype=np.uint8)
        gt[visual_field+tg_bbox[0]:visual_field+tg_bbox[2], visual_field+tg_bbox[1]:visual_field+tg_bbox[3]] = 1

        return gt

    def __load_stim(self, stim_path):
        """
        Load the search image i.e. the main stimuli in which the modle have to search for the target
        """

        stimuli = loadimg(stim_path, target_size=self.stim_shape[0:2], rev_img_flag=self.rev_img_flag)
        temp_stim = np.uint8(self.bg_value*np.ones((1, self.stim_shape[0]+2*(self.eye_res+self.corner_bias), self.stim_shape[1]+2*(self.eye_res+self.corner_bias), 3)))
        temp_stim[:, self.eye_res+self.corner_bias:self.stim_shape[0]+self.eye_res+self.corner_bias, self.eye_res+self.corner_bias:self.stim_shape[1]+self.eye_res+self.corner_bias, :] = np.copy(stimuli)
        stimuli = np.copy(temp_stim)
        return stimuli

    def reformat_gt_mask(self):
        """
        refer to __load_gt()
        """

        gt_mask_temp = []

        for i in range(self.gt_mask.shape[0]):
            gt = np.copy(self.gt_mask[i])
            gt = cv2.resize(gt, (self.stim_shape[1], self.stim_shape[0]), interpolation = cv2.INTER_AREA)
            retval, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
            temp_stim = np.uint8(np.zeros((self.stim_shape[0]+2*(self.eye_res+self.corner_bias), self.stim_shape[1]+2*(self.eye_res+self.corner_bias))))
            temp_stim[self.eye_res+self.corner_bias:self.stim_shape[0]+self.eye_res+self.corner_bias, self.eye_res+self.corner_bias:self.stim_shape[1]+self.eye_res+self.corner_bias] = np.copy(gt)
            gt = np.uint8(np.copy(temp_stim/255))

            gt_mask_temp.append(gt)

        self.gt_mask = np.array(gt_mask_temp)

    def post_process_saccades(self, saccade):
        """
        To perform the search expeirments, input images were padded to perform include both non-stimulus + stimulus region in the visual area.
        So after predicting the sacades, it changes the saccade relative to the input image.
        """

        if self.fix is not None:
            def get_pos(x, y, t):
                for i in range(self.gt_mask.shape[0]-1, -1, -1):
                    fxt_place = self.gt_mask[i][int(x), int(y)]
                    if (fxt_place > 0):
                        t = i + 1
                        break
                return t

            j = saccade.shape[0]
            for k in range(j):
                tar_id = get_pos(saccade[k, 0], saccade[k, 1], 0)
                saccade[k, 0] = self.fix[tar_id][0]
                saccade[k, 1] = self.fix[tar_id][1]
        else:
            saccade = saccade - self.eye_res - self.corner_bias

        return saccade
