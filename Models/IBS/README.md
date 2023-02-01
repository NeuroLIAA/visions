# VisualSearchPy

## Installation
```
git clone git@github.com:gastonbujia/VisualSearchPy.git
pip install -r ./VisualSearchPy
```
## Usage
### Runs visual search model on the whole dataset

```
python run_visualsearch.py
```

### Run with a different setup
```
python run_visualsearch.py --cfg greedy
```

Configuration files are located in /configs
### Run on a specific image or subset of images
```
python run_visualsearch.py --img grayscale_100_oliva.jpg
```
```
python run_visualsearch.py --rng 1 30
```
