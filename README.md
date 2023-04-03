# Uncertainty Quantification of the Black Hole Mass Estimation

Predicting virial black hole masses using neural network model and quantifying their uncertainties.
<!-- TODO: add link to paper -->
This repository is to accompany the paper Uncertainty Quantification of the Virial Black Hole Mass with Conformal Prediction. 

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
2. Uncertainty quantification for regression using MAPIE: `run_uqregressor.py`. Generated output files:
   - `estimator_optim.pkl`: Optimized GradientBoostingRegressor model for normal regression
   - `estimatorq_optim.pkl`: Optimized GradientBoostingRegressor model for quantile regression
   - `mapieuq_<STRATEGY>.pkl`: Fitted MapieRegressor for specified strategy
   - `mapieuq_pred.pkl`: y_pred, y_pis, sorted {'target', 'pred', 'lower', 'upper', 'pierr_metric'}
   - `mapieuq_picp_alpha.pkl`: DataFrame of prediction interval coverage probability (PICP) for different uncertainty quantification methods with alpha as index
   - `mapieuq_mpiw_alpha.pkl`: DataFrame of mean prediction interval width (MPIW) for different uncertainty quantification methods with alpha as index
3. Gradient boosting regressor for conformal prediction: `notebooks/conformalregressor.ipynb`

## Dataset
- The full catalogue of Sloan Digital Sky Survey (SDSS) DR16 quasar properties are available at: `http://quasar.astro.illinois.edu/paper_data/DR16Q/`
- To download data, run: `python get_data.py`
- Config file to specify the data directory: `config/config_getdata.yaml`
- Data exploratory notebook: `notebooks/inspect_data.ipynb`

## Example Exploratory Notebooks
- `notebooks/inspect_data.ipynb`: Inspect data from SDSS
- `notebooks/featureextraction_modelout.ipynb`: Examine outputs of feature extraction
- `notebooks/conformalregressor.ipynb`: Analysis of uncertainty quantification regressor

## Reproducible Results
The `reproducible_output/` directory contains the generated logs and output files that are used for the analysis in our paper.

## Citation
If you find this repository useful, please cite the paper:
```
TODO: insert citation
```

## References and Resources
- Taquet, Vianney, V. Blot, Thomas Morzadec, Louis Lacombe and Nicolas J.-B. Brunel. “MAPIE: an open-source library for distribution-free uncertainty quantification.” ArXiv abs/2207.12274 (2022). \[[Paper](https://arxiv.org/abs/2207.12274) | [Code](https://github.com/scikit-learn-contrib/MAPIE)\]
- Romano, Yaniv, Evan Patterson and Emmanuel J. Candès. “Conformalized Quantile Regression.” Neural Information Processing Systems (2019). \[[Paper](https://arxiv.org/abs/1905.03222) | [Code](https://github.com/yromano/cqr)\]
- Angelopoulos, Anastasios Nikolas and Stephen Bates. “A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.” ArXiv abs/2107.07511 (2021). \[[Paper](https://arxiv.org/abs/2107.07511) | [Code](https://github.com/aangelopoulos/conformal-prediction)\]

## TODO
- [ ] copy logs to reproducible_output
- [ ] add citations above
