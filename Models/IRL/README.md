# Scanpath Prediction Using Inverse Reinforcement Learning

PyTorch implementation of the article *[Yang, Z. et. al., Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Predicting_Goal-Directed_Human_Attention_Using_Inverse_Reinforcement_Learning_CVPR_2020_paper.html)*.
Official repository of this project can be found [here](https://github.com/cvlab-stonybrook/Scanpath_Prediction).

### Image preprocessing
The model requires images to be preprocessed by the pretrained Panoptic FPN (with ResNet50 backbone) from [Detectron2](https://github.com/facebookresearch/detectron2). In this implementation, this is done automatically by the script [```build_belief_maps.py```](irl_dcb/build_belief_maps.py).
