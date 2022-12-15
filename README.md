# ViSioNS: Visual Search in Natural Scenes Benchmark
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
| **nnIBS**   | **0.55**       | 0.84 | 0.15 | 0.74 | **1.27** | **0.44** | **0.35** | -0.17 |
| cIBS   | 0.51        | **0.85** | **0.17** | **0.75** | 1.26 | 0.31 | 0.23 | -0.19 |
| sIBS   | 0.54        | 0.84 | 0.13 | 0.74 | 1.25 | 0.31 | 0.23 | -0.19 |
| Center bias | - | - | - | 0.72 | 0.89 | 0.00 | 0.07 | -0.70 |
| Uniform | - | - | - | 0.50 | 0.00 | -0.07 | 0.00 | -0.87 |
| IVSN    | 0.67        | 0.80 | 0.09 | 0.61 | 1.07 | -4.29 | -4.18 | -0.91 |
| IRL     | 0.40       | 0.80 | 0.04 | 0.65 | 1.24 | -4.83 | -4.90 | -1.00 |

```AUCperf``` measures efficiency, while ```AvgMM``` is the Multi-Match average between models and subjects. ```Corr``` is the correlation between within-humans Multi-Match (whMM) and human-model Multi-Match (hmMM). Lastly, ```AUChsp```, ```NSShsp```, ```IGhsp``` and ```LLhsp``` focus on human scanpath prediction (HSP). The precise definition of each can be found at [```Metrics```](./Metrics). The scores for individual datasets can be found at their corresponding directory in [```Results```](./Results) (see ```Table.png```).

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
* ```--f```: Forces execution. Precomputed results are deleted and the specified models and metrics are run from scratch. WARNING: It will take a long time!

#### Command line example
Runs ```nnIBS``` on the ```COCOSearch18``` and ```Interiors``` datasets, and then computes Multi-Match and Cumulative Performance:
```
python3 run_benchmark.py --d COCOSearch18 Interiors --m nnIBS --mts mm perf
```
## How to cite us
If you use our work, please cite us:
```
@inproceedings{
  travi2022visions,
  title={ViSio{NS}: Visual Search in Natural Scenes Benchmark},
  author={Ferm{\'\i}n Travi and Gonzalo Ruarte and Gaston Bujia and Juan E Kamienkowski},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  pages={},
  url={https://openreview.net/forum?id=wPEXGTzZJt}
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
