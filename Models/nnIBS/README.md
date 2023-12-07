# Modeling human visual search: A combined Bayesian searcher and saliency map approach for eye movement guidance in natural scenes

Official Python implementation of the model described in [Bujia G., Sclar M., Vita S., Solovey G., Kamienkowski JE., *Modeling Human Visual Search in Natural Scenes: A Combined Bayesian Searcher and Saliency Map Approach*, Frontiers in Systems Neuroscience, Volume 16, 2022.](https://www.frontiersin.org/article/10.3389/fnsys.2022.882315). Original MATLAB code can be found [here](https://github.com/gastonbujia/VisualSearch).

### Template Response
This implementation allows for several possible variations when computing the ```templateResponse``` component described in the paper (here named ```target_similarity_map```). In particular, in addition to ```cross-correlation```, ```SSIM``` and ```IVSN``` are available. The latter is the default and consists of using [IVSN's attention map](../IVSN/ivsn_model/IVSN.py), which enables the model to generalize beyond cropped targets and perform object-invariant visual search.

To use another variation, simply change the ```target_similarity``` field in [configs/default.json](configs/default.json) to ```ssim``` or ```correlation```. 

### Prior
The model's prior is based on a saliency map computed by [DeepGazeIIE](https://github.com/matthias-k/DeepGaze) with a center bias template that must be downloaded from [here](https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy) and placed inside ```visualsearch/utils/deepgaze```.
