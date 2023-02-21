# Black Hole Mass Estimation

## Requirement
```
pip install -r requirement.txt
```

## Pipeline
0. Get data: `get_data.py`. See [Dataset](#dataset) on the catalogue and data used.
1. Feature extractor using fully-connected neural network model: `run_featureextractor.py`.

2. Gradient boosting regressor for conformal prediction: `notebooks/conformalregressor.ipynb`


## Dataset

The full catalogue of SDSS DR16 quasar properties are available at: `http://quasar.astro.illinois.edu/paper_data/DR16Q/`

To download data, run: `python get_data.py`

Config file to specify the data directory: `config/config_getdata.yaml`

Data exploratory notebook: `notebooks/inspect_data.ipynb`

