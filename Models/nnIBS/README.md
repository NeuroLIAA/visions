# Modeling human visual search: A combined Bayesian searcher and saliency map approach for eye movement guidance in natural scenes

Official Python implementation of the model described in *[Sclar, M., Bujia, G., Vita, S., Solovey, G., Kamienkowski, JE., Modeling human visual search: A combined Bayesian searcher and saliency map approach for eye movement guidance in natural scenes, arXiv preprint arXiv:2009.08373, 2020](https://arxiv.org/pdf/2009.08373.pdf)*. Original MATLAB code can be found [here](https://github.com/gastonbujia/VisualSearch).

### Template Response
This implementation allows for several possible variations when computing the ```templateResponse``` component described in the paper (here named ```target_similarity_map```). In particular, in addition to ```cross-correlation```, ```SSIM``` and ```IVSN``` are available. The latter is the default and consists of using [IVSN's attention map](../IVSN/ivsn_model/IVSN.py), which enables the model to generalize beyond cropped targets and perform object-invariant visual search.

To use another variation, simply change the ```target_similarity``` field in [configs/default.json](configs/default.json) to ```ssim``` or ```correlation```. 
