import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.special import logsumexp
from skimage import io, color, img_as_ubyte
import sys
from .deepgaze_pytorch import DeepGazeIIE
import torch
from pathlib import Path
DEVICE = 'cpu'
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def create_saliencymap_for_image(image, save_path):
    print('Creating saliency map for search image...')
    model = DeepGazeIIE(pretrained=True).to(DEVICE)
    script_path = Path(__file__).resolve().parent
    centerbias_template = np.load(script_path / 'centerbias.npy')
    # rescale to match image size
    centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    # centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]
    
    if len(image.shape) < 3:
        image = color.gray2rgb(image)
    # image_data = image[np.newaxis, :, :, :] 

    image_tensor = torch.tensor(image.transpose(2, 0, 1)[np.newaxis, :]).to(DEVICE)
    centerbias_tensor = torch.tensor(centerbias[np.newaxis, :]).to(DEVICE)

    log_density_prediction = model(image_tensor, centerbias_tensor).detach().cpu().numpy()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(save_path, np.exp(log_density_prediction[0, 0, :, :]), cmap=plt.cm.gray)
    # Remove channels and keep one
    image = io.imread(save_path)
    io.imsave(save_path, img_as_ubyte(color.rgb2gray(image)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '-img_path', type=str, help='Path to the image file on which to run DeepGaze II')
    parser.add_argument('--o', '-output_path', type=str, default='./', help='Path where the saliency map will be saved')

    args = parser.parse_args()
    if not Path(args.img).is_file():
        print('Wrong path to image file')
        sys.exit(-1)

    image = io.imread(args.img)
    image_name  = str(args.img).split('/')[-1]
    output_path = Path(args.o)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f'{image_name[:-4]}_saliency{image_name[-4:]}'

    create_saliencymap_for_image(image, output_path)    
