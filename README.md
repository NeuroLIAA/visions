# Visual Search in natural scenes Benchmark

## Ranking
Current models' score, relative to human subjects, averaged across all datasets:
|       | AUCperf | AvgMM | AUChsp | NSShsp | Score |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **nnIBS**   | **0.867**       | **0.843** | **0.737** | **1.12** | **0.892** |
| **IRL**     | 0.836       | 0.798 | 0.644 | 1.102 | 0.845 |
| **IVSN**    | 0.851        | 0.802 | 0.606 | 0.835 | 0.774 |

```AUCperf``` measures efficiency, while ```AvgMM```, ```AUChsp``` and ```NSShsp``` focus on scanpath similarity and prediction, respectively. The precise definition of each one can be found at [```Metrics```](./Metrics). The scores for each independent dataset can be found at its corresponding directory in [```Results```](./Results).

## Installation
Python 3 is required. To install, run:
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
* ```--d dataset1_name dataset2_name .. ```: runs the visual search models in the datasets specified. Each dataset name corresponds to its folder's name in [```Datasets```](./Datasets).
* ```--m model1_name model2_name .. ```: runs the specified visual search models. Each model name corresponds to its folder's name in [```Models```](./Models).
* ```--mts [mm] [perf] [hsp] ```: computes the specified metrics. Values can be ```mm``` (Multi-Match), ```perf``` (Cumulative Performance), and/or ```hsp``` (Human Scanpath Prediction). See [```Metrics```](./Metrics) for more information. Leave blank to not compute any metric.
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
travi2021benchmarking,
title={Benchmarking human visual search computational models in natural scenes: models comparison and reference datasets},
author={Ferm{\'\i}n Travi and Gonzalo Ruarte and Gaston Bujia and Juan E Kamienkowski},
booktitle={SVRHM 2021 Workshop @ NeurIPS },
year={2021},
url={https://openreview.net/forum?id=ng262VIrK08}
}
```
```
@inproceedings{
sclar2020modeling,
title={Modeling human visual search: A combined Bayesian searcher and saliency map approach for eye movement guidance in natural scenes},
author={Melanie Sclar and Gaston Bujia and Sebastian Vita and Guillermo Solovey and Juan Esteban Kamienkowski},
booktitle={NeurIPS 2020 Workshop SVRHM},
year={2020},
url={https://openreview.net/forum?id=e35q2TmbZbw}
}
```

The exact materials used for the paper can be found at the [SVRHM branch](https://github.com/FerminT/VisualSearchBenchmark/tree/SVRHM).
