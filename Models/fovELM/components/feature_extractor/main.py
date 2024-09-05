import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Pad

NAMED_LABELS = [
    "",
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "trafficlight",
    "firehydrant",
    "streetsign",
    "stopsign",
    "parkingmeter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eyeglasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sportsball",
    "kite",
    "baseballbat",
    "baseballglove",
    "skateboard",
    "surfboard",
    "tennisracket",
    "bottle",
    "plate",
    "wineglass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "mirror",
    "diningtable",
    "window",
    "desk",
    "toilet",
    "door",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cellphone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddybear",
    "hairdrier",
    "toothbrush",
    "hairbrush",
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(NAMED_LABELS)}

DEFAULT_PADDING = 115


class FeatureExtractor:
    """Feature extractor for the images
    Extracts features from the images using a Faster R-CNN model
    Can also contruct a P map from the bounding boxes and scores
    Parameters:
        images: Array of images
        target: Target category for constructing the foveal p map
        config: Configuration for the feature extractor
            stride: Stride set for the p_map
    """

    def __init__(self, images: np.ndarray, target: str, config: dict[str, any]):
        self.images = images
        self.image_features = None
        self.target = target
        self.config = config
        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.eval()

        self.extractor = create_feature_extractor(self.model.backbone, {"fpn": "fpn"})
        self.transformed_images = self.transform_images(images)

    def transform_images(self, images: np.ndarray) -> np.ndarray:
        """Transform images to the correct format for the model
        Args:
            images: Array of images
        Returns:
            Transformed images
        """
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        transforms = weights.transforms()
        padder = Pad(padding=DEFAULT_PADDING, fill=0)
        transposed_images = images.transpose((0, 3, 1, 2))
        padded_images = padder(torch.from_numpy(transposed_images))
        return transforms(padded_images)

    def extract(self) -> np.ndarray:
        """Extract features from images
        Returns 4 feature maps:
            - P0: 1/4 of the input size
            - P1: 1/8 of the input size
            - P2: 1/16 of the input size
            - P3: 1/32 of the input size
        """
        if self.image_features:
            return dict(self.image_features["fpn"])
        self.image_features = self.extractor(self.transformed_images)
        return dict(self.image_features["fpn"])

    def construct_p_map(self, boxes, scores) -> np.ndarray:
        """Construct a P map from the bounding boxes and scores
        Args:
            boxes: List of bounding boxes
            scores: List of scores
        Returns:
            p_map for the grid locations of the image using the stride set in the config
            If center of bounding box is in the grid location, the score is set for that location
        """
        p_map = np.zeros(self.transformed_images.shape[2:])
        stride = self.config["stride"]
        grid_locations = np.array(
            [
                (x + DEFAULT_PADDING, y + DEFAULT_PADDING)
                for x in range(0, p_map.shape[1], stride)
                for y in range(0, p_map.shape[0], stride)
            ]
        )

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.detach().numpy()
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            closest_grid = grid_locations[
                np.argmin(np.linalg.norm(grid_locations - (x_center, y_center), axis=1))
            ]
            
            p_map[
                closest_grid[1] : closest_grid[1] + stride,
                closest_grid[0] : closest_grid[0] + stride,
            ] = max(score.detach().numpy(), p_map[closest_grid[1], closest_grid[0]])
        return p_map

    # TODO: check if p_maps should keep transformed size
    def get_foveal_p_map(self) -> np.ndarray:
        """Get the foveal P map for the images of the target object
        Runs the entire image through the model and extracts the bounding boxes and predictions
        Keeps only the bounding boxes that contain the target object
        """
        with torch.no_grad():
            predictions = self.model(self.transformed_images)
        print(predictions)
        target_idx = LABEL_TO_IDX[self.target]

        results = []
        for prediction in predictions:
            target_boxes = [
                box
                for box, label in zip(prediction["boxes"], prediction["labels"])
                if label == target_idx
            ]
            target_scores = [
                score
                for score, label in zip(prediction["scores"], prediction["labels"])
                if label == target_idx
            ]
            results.append(self.construct_p_map(target_boxes, target_scores))
        return np.array(results)
