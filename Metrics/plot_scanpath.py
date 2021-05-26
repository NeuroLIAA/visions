import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import argparse
import json

datasets_dir = '../Datasets/'
results_dir  = '../Results/'

def plot_scanpath(img, xs, ys, bbox=None, title=None):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)

    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1], ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)

    for i in range(len(xs)):
        circle = plt.Circle((xs[i], ys[i]),
                            radius=30,
                            edgecolor='red',
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(i + 1), xy=(xs[i], ys[i] + 3), fontsize=10, ha="center", va="center")

    if bbox is not None:
        rect = Rectangle((bbox[1], bbox[0]), bbox[2], bbox[3], alpha=0.5, edgecolor='yellow', facecolor='none', linewidth=2)
        ax.add_patch(rect)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='Name of the visual search model')
    parser.add_argument('-dataset', type=str, help='Name of the dataset')
    parser.add_argument('-img', type=str, help='Name of the image on which to draw the scanpath')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    scanpaths_file = results_dir + args.dataset + '_dataset/' + args.model + '/Scanpaths.json'
    with open(scanpaths_file, 'r') as fp:
        scanpaths = json.load(fp)

    img_scanpath = scanpaths[args.img]
    X = img_scanpath['X']
    Y = img_scanpath['Y']

    bbox = img_scanpath['target_bbox']
    target_height = bbox[2] - bbox[0]
    target_width  = bbox[3] - bbox[1]
    bbox = [bbox[0], bbox[1], target_width, target_height]
    
    # TODO: Levantar del JSON del dataset
    if args.dataset == 'cIBS':
        image_folder = 'images'
    else:
        image_folder = 'stimuli'

    image_file = datasets_dir + args.dataset + '/' + image_folder + '/' + args.img
    img = mpimg.imread(image_file)

    title = args.model + ' ' + args.img
    plot_scanpath(img, X, Y, bbox, title)