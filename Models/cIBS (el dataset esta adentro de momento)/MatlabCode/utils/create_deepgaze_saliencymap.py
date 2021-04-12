import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.special import logsumexp
from skimage import io, color
from os import listdir, mkdir

import tensorflow.compat.v1 as tf
# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
# Ignore warnings
tf.logging.set_verbosity(tf.logging.WARNING)

img_size = (768, 1024)
# load precomputed log density over a 1024x1024 image
centerbias_template = np.load('centerbias.npy')  
# rescale to match image size
centerbias = zoom(centerbias_template, (img_size[0]/1024, img_size[1]/1024), order=0, mode='nearest')
# renormalize log density
centerbias -= logsumexp(centerbias)

centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)

tf.reset_default_graph()
check_point = 'DeepGazeII.ckpt'  # DeepGaze II
#check_point = 'ICF.ckpt'  # ICF
new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))

input_tensor = tf.get_collection('input_tensor')[0]
centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
log_density = tf.get_collection('log_density')[0]
log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]

images_path = '../data_images/images/'
save_path = '../saliency/deepgaze/'
images = listdir(images_path)
for image_name in images:
    img = io.imread(images_path + image_name)
    img = color.gray2rgb(img)
    image_data = img[np.newaxis, :, :, :]  # BHWC, three channels (RGB)

    with tf.Session() as sess:
        new_saver.restore(sess, check_point)
        
        log_density_prediction = sess.run(log_density, {
            input_tensor: image_data,
            centerbias_tensor: centerbias_data,
        })

    plt.imsave(save_path + image_name, np.exp(log_density_prediction[0, :, :, 0]), cmap=plt.cm.gray)
