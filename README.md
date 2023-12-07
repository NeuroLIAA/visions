# ViSioNS: Visual Search in Natural Scenes Benchmark
## Updates
* **People dataset added to the benchmark!** (Results will be uploaded soon)
  * This dataset dates back to 2009 and belongs to the paper [*Ehinger, K. A., Hidalgo-Sotelo, B., Torralba, A., & Oliva, A, "Modeling Search for People in 900 Scenes: A combined source model of eye guidance", Visual cognition, 17(6-7), 945–978, 2009*.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2790194/)
* **eccNET added to the benchmark!**
  * This visual search model belongs to the paper [*Gupta SK., Zhang M., Wu C., Wolfe JM., Kreiman G., "Visual Search Asymmetry: Deep Nets and Humans Share Similar Inherent Biases", NeurIPS 2021*](https://github.com/kreimanlab/VisualSearchAsymmetry).
## About
Visual search is an essential part of almost any everyday human interaction with the visual environment. Nowadays, several algorithms are able to predict gaze positions during simple observation, but few models attempt to simulate human behavior during visual search in natural scenes. Furthermore, these models vary widely in their design and exhibit differences in the datasets and metrics with which they were evaluated.

To attend to this problem, we have selected publicly available state-of-the-art visual search models and datasets in natural scenes, and provide a common framework for their evaluation. We apply a unified format and criteria, bridging the gaps between them, and we estimate the models’ efficiency and similarity with humans using a specific set of metrics.

![Scanpath example](Metrics/Plots/Scanpath_example.png#gh-light-mode-only)
![Scanpath example](Metrics/Plots/Scanpath_example_white.png#gh-dark-mode-only)

## Ranking
Current models scores, relative to human subjects, averaged across all datasets:
|       | AUC*perf* | AvgMM | Corr | AUC*hsp* | NSS*hsp* | IG*hsp* | LL*hsp* | Score |
| ----------- | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| Humans | 0.56 | 0.87 | - | - | - | - | - | - | - |
| Gold Standard | - | - | - | 0.90 | 2.65 | 1.93 | 1.95 | 0.00 |
| **nnIBS**   | 0.48 | **0.85** | **0.25** | **0.79** | **1.50** | **0.57** | **0.49** | -0.12 |
| eccNET   | **0.52** | 0.84 | 0.19 | 0.66 | 1.03 | -2.20 | -2.01 | -0.57 |
| Center bias | - | - | - | 0.72 | 0.89 | 0.00 | 0.07 | -0.70 |
| Uniform | - | - | - | 0.50 | 0.00 | -0.07 | 0.00 | -0.87 |
| IVSN    | 0.67        | 0.80 | 0.09 | 0.61 | 1.07 | -4.29 | -4.18 | -0.91 |
| IRL     | 0.40       | 0.80 | 0.04 | 0.65 | 1.24 | -4.83 | -4.90 | -1.00 |

```AUCperf``` measures efficiency, while ```AvgMM``` is the Multi-Match average between models and subjects. ```Corr``` is the correlation between within-humans Multi-Match (whMM) and human-model Multi-Match (hmMM). Lastly, ```AUChsp```, ```NSShsp```, ```IGhsp``` and ```LLhsp``` focus on human scanpath prediction (HSP). The precise definition of each can be found at [```Metrics```](./Metrics).

**Scores for individual datasets:**
* [```Interiors```](./Results/Interiors/Table.png)
* [```Unrestricted```](./Results/Unrestricted/Table.png)
* [```COCOSearch18```](./Results/COCOSearch18/Table.png)
* [```MCS```](./Results/MCS/Table.png)

## Installation
Python 3.8 or newer is required. To install, run:
```
git clone git@github.com:FerminT/VisualSearchBenchmark.git
pip3 install -r ./VisualSearchBenchmark/requirements.txt
```

```Detectron2``` needs to be installed separately (see [detectron2/installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)). It is only needed if you're going to run the IRL model on new images.
## Usage
Runs all models in every dataset and compute all metrics (by default, precomputed results are loaded):
```
python3 run_benchmark.py
```
The code was tested in Ubuntu 16.04 and later.
#### Optional parameters
* ```--d dataset1_name dataset2_name ..```: runs the visual search models in the datasets specified. Each dataset name corresponds to its folder's name in [```Datasets```](./Datasets).
* ```--m model1_name model2_name ..```: runs the specified visual search models. Each model name corresponds to its folder's name in [```Models```](./Models).
* ```--mts [mm] [perf] [hsp]```: computes the specified metrics. Values can be ```mm``` (Multi-Match), ```perf``` (Cumulative Performance), and/or ```hsp``` (Human Scanpath Prediction). See [```Metrics```](./Metrics) for more information. Leave blank to not compute any metric.
* ```--noplot```: Do not plot the results. Useful for running the benchmark and going AFK.
* ```--f```: Forces execution. Precomputed results are deleted and the specified models and metrics are run from scratch. WARNING: it will take a long time!

#### Command line example
Runs ```nnIBS``` on the ```COCOSearch18``` and ```Interiors``` datasets, and then computes Multi-Match and Cumulative Performance:
```
python3 run_benchmark.py --d COCOSearch18 Interiors --m nnIBS --mts mm perf
```
## How to cite us
If you use our work, please cite us:
```
@inproceedings{travi2022visions,
 author = {Travi, Ferm\'{\i}n and Ruarte, Gonzalo and Bujia, Gaston and Kamienkowski, Juan Esteban},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {11987--12000},
 publisher = {Curran Associates, Inc.},
 title = {ViSioNS: Visual Search in Natural Scenes Benchmark},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/4ea14e6090343523ddcd5d3ca449695f-Paper-Datasets_and_Benchmarks.pdf},
 volume = {35},
 year = {2022}
}
```
```
@article{Bujia2022vsearch,
author = {Bujia, Gaston and Sclar, Melanie and Vita, Sebastian and Solovey, Guillermo and Kamienkowski, Juan Esteban},
doi = {10.3389/fnsys.2022.882315},
issn = {1662-5137},
journal = {Frontiers in Systems Neuroscience},
title = {Modeling Human Visual Search in Natural Scenes: A Combined Bayesian Searcher and Saliency Map Approach},
url = {https://www.frontiersin.org/article/10.3389/fnsys.2022.882315},
volume = {16},
year = {2022}
}
```

The exact materials used for the paper can be found at the [1. NeurIPS 2022 branch](https://github.com/FerminT/VisualSearchBenchmark/tree/NeurIPS), [2. SVRHM 2021 branch](https://github.com/FerminT/VisualSearchBenchmark/tree/SVRHM), [3. Frontiers in Systems Neuroscience repository](https://github.com/gastonbujia/VisualSearch). 

## References
### Models
* eccNET: [Gupta SK., Zhang M., Wu C., Wolfe JM., Kreiman G., *Visual Search Asymmetry: Deep Nets and Humans Share Similar Inherent Biases*, NeurIPS 2021.](https://github.com/kreimanlab/VisualSearchAsymmetry)
* IVSN: [Zhang M. et al., *Finding any Waldo with Zero-shot Invariant and Efficient Visual Search Model*, Nature Comun. 9, 2018.](https://www.nature.com/articles/s41467-018-06217-x)
* nnIBS: [Bujia G., Sclar M., Vita S., Solovey G., Kamienkowski JE., *Modeling Human Visual Search in Natural Scenes: A Combined Bayesian Searcher and Saliency Map Approach*, Frontiers in Systems Neuroscience, Volume 16, 2022.](https://www.frontiersin.org/article/10.3389/fnsys.2022.882315)
* IRL: [Yang, Z. et. al., *Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning*, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Predicting_Goal-Directed_Human_Attention_Using_Inverse_Reinforcement_Learning_CVPR_2020_paper.html)
### Datasets
* Interiors: [Bujia G., Sclar M., Vita S., Solovey G., Kamienkowski JE., *Modeling Human Visual Search in Natural Scenes: A Combined Bayesian Searcher and Saliency Map Approach*, Frontiers in Systems Neuroscience, Volume 16, 2022.](https://www.frontiersin.org/article/10.3389/fnsys.2022.882315)
* Unrestricted: [Zhang M. et al., *Finding any Waldo with Zero-shot Invariant and Efficient Visual Search Model*, Nature Comun. 9, 2018.](https://www.nature.com/articles/s41467-018-06217-x)
* COCOSearch18: [Chen, Y., Yang, Z., Ahn, S., Samaras, D., Hoai, M., & Zelinsky, G., *COCO-Search18 Fixation Dataset for Predicting Goal-directed Attention Control*, Scientific Reports, 11 (1), 1-11, 2021.](https://www.nature.com/articles/s41598-021-87715-9)
* MCS: [G.J. Zelinsky et al., *Benchmarking Gaze Prediction for Categorical Visual Search*, CVPR Workshops 2019, 2019.](https://www3.cs.stonybrook.edu/~zhibyang/papers/Gaze_Benchmark_CVPRw.pdf)
* People: [Ehinger, K. A., Hidalgo-Sotelo, B., Torralba, A., & Oliva, A, *Modeling Search for People in 900 Scenes: A combined source model of eye guidance*, Visual cognition, 17(6-7), 945–978, 2009.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2790194/)