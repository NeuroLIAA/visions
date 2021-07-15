from .target_similarity import TargetSimilarity

""" Target similarity is computed as in Geisler et. al. (2005). That is to say, no further calculations are made """

class Geisler(TargetSimilarity):
    def compute_target_similarity(self, image, target, target_bbox):
        return None
    
    def add_info_to_mu(self, target_similarity_info, visibility_map):
        return 