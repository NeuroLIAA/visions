# Visual Search in natural scenes Benchmark

## Installation
Python 3 is required. To install, run:
```
git clone git@github.com:FerminT/VisualSearchBenchmark.git
pip3 install -r ./VisualSearchBenchmark
```

## Usage
Runs all models in every dataset and compute all metrics (by default, precomputed results are loaded):
```
python3 run_benchmark.py
```

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
Travi, F., Ruarte, G., Bujia, G. & Kamienkowski, J. E. (2021). Benchmarking human visual search computational models: models comparison and reference datasets. *In Shared Visual Representations in Human & Machine intelligence, 2021 NeurIPS workshop.*

The exact materials used for the paper can be found at the [SVRHM branch](https://github.com/FerminT/VisualSearchBenchmark/tree/SVRHM)
