# Visual Search in natural scenes Benchmark

## Installation
Python 3 is required. To install, run:
```
git clone git@github.com:FerminT/VisualSearchBenchmark.git
pip3 install -r ./VisualSearchBenchmark
```
## Usage
Run all models in every dataset and compute all metrics (by default, precomputed results are loaded):
```
python3 run_benchmark.py
```
#### Optional parameters
* ```--d dataset1_name dataset2_name .. ```: runs the visual search models in the datasets specified. Each dataset name corresponds to its folder's name in ```Datasets```.
* ```--m model1_name model2_name .. ```: runs the specified visual search models. Each model name corresponds to its folder's name in ```Models```.
* ```--mts [mm] [perf] [hsp] ```: computes the specified metrics. Values can be ```mm``` (Multi-Match), ```perf``` (Cumulative Performance), and/or ```hsp``` (Human Scanpath Prediction). See ```Metrics``` for more information.
* ```--f```: Forces execution. Precomputed results are deleted and the specified models and metrics are run from scratch. WARNING: It will take a long time!
