from .target_similarity import TargetSimilarity
from skimage.feature import match_template

""" Target similarity is computed via normalized cross correlation """

class Correlation(TargetSimilarity):
    def compute_target_similarity(self, image, target, target_bbox):
        """ Input:
                image  (2D array) : image where the target is
                target (2D array) : target image      
            Output:
                cross_correlation (2D array) : target similarity map using normalized cross correlation
        """
        cross_correlation = match_template(image, target, pad_input=True)
        if len(cross_correlation.shape) > 2:
            # If it's coloured, convert to single channel using skimage rgb2gray formula
            cross_correlation = cross_correlation[:, :, 0] * 0.2125 + \
                                cross_correlation[:, :, 1] * 0.7154 + \
                                cross_correlation[:, :, 2] * 0.0721
        
        return cross_correlation