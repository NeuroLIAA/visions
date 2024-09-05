import numpy as np
from torch import nn

DEFAULT_PADDING = 115


def max_pooler(kernel_size: int, stride: int):
    """Create a max pooling layer with the given kernel size and stride"""
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)


# TODO: Check if the amount of extracted feature vectors is correct according to stride
# should be the same as the amount of grid locations in the foveal_p_map of the FeatureExtractor
def eccentricity_pooler(feature_columns: np.ndarray, stride: int):
    """Eccentricity based max pooling
    Args:
        feature_columns: np.ndarray of shape (N, C, H, W). Has padding of DEFAULT_PADDING // 4 cause of the model
        stride: int, stride of the sampling of the feature columns
    Output:
        List of results for each configuration of the eccentricity based max pooling
        0: feature vectors of size 1024
        1, 2, 3: feature vectors of size 256
    """
    configurations = [(25, 15, 10), (35, 35, 1), (45, 45, 1), (55, 55, 1)]
    padding = DEFAULT_PADDING // 4
    grid_locations = np.array(
        [
            (x + stride // 2, y + stride // 2)
            for x in range(padding, feature_columns.shape[3] - padding - stride, stride)
            for y in range(padding, feature_columns.shape[2] - padding - stride, stride)
        ]
    )

    print(feature_columns.shape, stride)
    results = []
    for config in configurations:
        sample_size, kernel_size, stride = config
        pooler = max_pooler(kernel_size, stride)

        config_results = []
        for x, y in grid_locations:
            sample = feature_columns[
                :,
                :,
                y - sample_size // 2 : y + sample_size // 2 + 1,
                x - sample_size // 2 : x + sample_size // 2 + 1,
            ]
            pooled_sample = pooler(sample)
            flattened_sample = pooled_sample.flatten()
            config_results.append(flattened_sample)

        results.append(config_results)

    return results
