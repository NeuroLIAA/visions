import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.special import logsumexp
from skimage import io, color, img_as_ubyte
from os import listdir, mkdir, path, environ, getcwd
# Ignore tensorflow messages
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow.compat.v1 as tf

def create_saliencymap_for_image(image, save_path):
    # Ignore warnings
    tf.logging.set_verbosity(tf.logging.ERROR)
    # To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
    tf.disable_eager_execution()

    # load precomputed log density over a 1024x1024 image
    script_path = path.dirname(__file__)
    centerbias_template = np.load(path.join(script_path, 'centerbias.npy'))
    # rescale to match image size
    img_size   = (image.shape[0], image.shape[1])
    centerbias = zoom(centerbias_template, (img_size[0]/1024, img_size[1]/1024), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)

    centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)

    tf.reset_default_graph()
    check_point = 'DeepGazeII.ckpt'  # DeepGaze II
    new_saver   = tf.train.import_meta_graph(path.join(script_path, '{}.meta'.format(check_point)))

    input_tensor      = tf.get_collection('input_tensor')[0]
    centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
    log_density       = tf.get_collection('log_density')[0]

    if len(image.shape) < 3:
        image = color.gray2rgb(image)
    image_data = image[np.newaxis, :, :, :]  # BHWC, three channels (RGB)

    with tf.Session() as sess:
        new_saver.restore(sess, path.join(script_path, check_point))
        
        log_density_prediction = sess.run(log_density, {
            input_tensor: image_data,
            centerbias_tensor: centerbias_data,
        })

    plt.imsave(save_path, np.exp(log_density_prediction[0, :, :, 0]), cmap=plt.cm.gray)
    # Remove channels and keep one
    image = io.imread(save_path)
    io.imsave(save_path, img_as_ubyte(color.rgb2gray(image)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '-img_path', type=str, help='Path to the image file on which to run DeepGaze II')
    parser.add_argument('--o', '-output_path', type=str, default='./', help='Path where the saliency map will be saved')

    args = parser.parse_args()

    if not path.isfile(args.img):
        print('Wrong path to image file')
        sys.exit(-1)

    image = io.imread(args.img)
    image_name  = str(args.img).split('/')[-1]
    output_path = path.join(args.o, '') + image_name[:-4] + '_saliency' + image_name[-4:]

    create_saliencymap_for_image(image, output_path)    
