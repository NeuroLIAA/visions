# VisualSearch
This repository contains the code for the paper [*Modeling human visual search: A combined Bayesian searcher and saliency map approach for eye movement guidance in natural scenes*](https://arxiv.org/pdf/2009.08373) presented in [Shared Visual Representations in Human & Machine Intelligence,
2020 NeurIPS Workshop (Dec 12, 2020)](https://www.svrhm.com/)

Please cite with the following BibTeX: 

@article{sclar2020modeling,
  title={Modeling human visual search: A combined Bayesian searcher and saliency map approach for eye movement guidance in natural scenes},
  author={Sclar, Melanie and Bujia, Gast{\'o}n and Vita, Sebasti{\'a}n and Solovey, Guillermo and Kamienkowski, Juan Esteban},
  journal={arXiv preprint arXiv:2009.08373},
  year={2020}
}

# Environment
The model was tested on Matlab 2016 and higher.

# Run Searcher in Matlab
To run the IBS model or any of the variants, run the *run\_main.m* script inside the dynamic_model folder (in MatlabCode folder). This script allows to set different search strategies and priors, the possible parameters (check BayesianSaliencyModel.m for more info) are:

* **Searcher**
	1. Correlation IBS 
	2. Geisler's IBS
	3. Greedy
* **Prior**
	1. DeepGaze2
	2. MLNet
	3. SAM-ResNet
	4. SAM-VGG
	5. Center
	6. Flat
	7. Noisy

For more info about the parameters check *BayesianSaliencyModel.m*.

## Use Searchers with different datasets

## TODO MATLAB CODE
- Add analysis code
- Update readme
- Add images to gitignore
