# Black Hole Mass Estimation

## Requirement
```
pip install -r requirement.txt
```

## Pipeline
0. Get data: `get_data.py`. See [Dataset](#dataset) on the catalogue and data used.
1. Feature extractor using fully-connected neural network model: `run_featureextractor.py`. Generated output files:
   - `datasplitidx.pkl`: dict of data split indices {'train' train_idx, 'test': test_idx}
   - `yscaler.pkl`: sklearn.preprocessing scaler
   - `model.pth`: Trained PyTorch model
   - `loss.pkl`: DataFrame of 'train' and 'test' losses
   - `features.pkl`: DataFrame of features, 'objid', 'label', 'scaled_label', 'output'
2. Uncertainty quantification for regressor using MAPIE: `run_uqregressor.py`. Generated output files:
   - `estimator_optim.pkl`: Optimized RandomForestRegressor model
   - `estimatorq_optim.pkl`: Optimized GradientBoostingRegressor model
   - `mapieuq_<STRATEGY>.pkl`: Fitted MapieRegressor for specified strategy
   - `mapieuq_pred.pkl`: y_pred, y_pis, {'target', 'pred', 'lower', 'upper', 'pierr_metric'}
   - `mapieuq_coverage_alpha.pkl`: DataFrame of coverage for different uncertainty quantification methods with alpha as index
   - `mapieuq_width_alpha.pkl`: DataFrame of width for different uncertainty quantification methods with alpha as index
3. Gradient boosting regressor for conformal prediction: `notebooks/conformalregressor.ipynb`


## Dataset

The full catalogue of SDSS DR16 quasar properties are available at: `http://quasar.astro.illinois.edu/paper_data/DR16Q/`

To download data, run: `python get_data.py`

Config file to specify the data directory: `config/config_getdata.yaml`

Data exploratory notebook: `notebooks/inspect_data.ipynb`

