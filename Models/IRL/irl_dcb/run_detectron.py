# import some common libraries
import numpy as np
import os, sys, cv2
import argparse
# Ignore deprecated warning
import warnings
warnings.simplefilter('ignore', UserWarning)

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

def run_detectron(image, output_shape=None, visualize=False):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    cfg.MODEL.DEVICE  = 'cpu'

    predictor = DefaultPredictor(cfg)
    panoptic_seg, segments_info = predictor(np.array(image))["panoptic_seg"]

    if visualize and (height, width) == image.shape[:2]:
        v   = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

        cv2.imshow('image', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    return panoptic_seg, segments_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '-img_path', type=str, help='Path to the image file on which to run Detectron')
    parser.add_argument('--size', '--size', nargs=2, default=None, type=int, help='Output size (height width)')

    args = parser.parse_args()

    if not os.path.isfile(args.img):
        print('Wrong path to file')
        sys.exit(0)

    image = cv2.imread(args.img)
    image = cv2.resize(image, (512, 320))

    run_detectron(image, args.size, visualize=True)