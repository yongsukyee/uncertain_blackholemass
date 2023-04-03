##################################################
# RUN UNCERTAINTY QUANTIFICATION FOR REGRESSION WITH MAPIE
# Author: Suk Yee Yong
##################################################


from mapie.metrics import regression_coverage_score, regression_mean_width_score
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import time


def optimize_regressor(X_train, y_train, regressor=None, n_iter=100, cv=10, filename='estimator_optim.pkl'):
    """Optimize regressor. Save output file: estimator_optim.pkl"""
    t1 = time.time()
    print(f"OPTIMIZING REGRESSOR ...")
    
    try:
        estimator = pd.read_pickle(filename)
        print(f"Loaded optimized regressor >> {filename}")
    except:
        param_distributions = {
            'learning_rate': stats.uniform(),
            'n_estimators': stats.randint(10, 500),
            'max_depth': stats.randint(2, 30),
            'max_leaf_nodes': stats.randint(2, 50),
        }
        gbr_kwargs = dict(loss='squared_error')
        if regressor == 'quantile':
            gbr_kwargs = dict(loss='quantile', alpha=0.5)
        estimator = GradientBoostingRegressor(random_state=cfg['seed'], verbose=0, **gbr_kwargs)
        # Search CV
        cv_obj = RandomizedSearchCV(
            estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            verbose=0,
            random_state=cfg['seed'],
            n_jobs=-1,
        )
        cv_obj.fit(X_train, y_train)
        estimator = cv_obj.best_estimator_
        pd.to_pickle(estimator, filename)
        print(f"\tTime >> {time.time() - t1}")
        print(f"Best estimator >> {estimator}")
    return estimator


def mapie_uq(estimator, estimator_q, X_train, y_train, X_calib, y_calib, X_test, y_test, list_strategies=None, alpha=0.1, filepath='./', save_outfile=True):
    """Compute different MAPIE uncertainty quantification methods for regression. Save output file: mapieuq_pred.pkl"""
    
    strategies_dict = {
        'naive': {'method': 'naive'},
        'jackknife': {'method': 'base', 'cv': -1},
        'jackknife_plus': {'method': 'plus', 'cv': -1},
        'jackknife_minmax': {'method': 'minmax', 'cv': -1},
        'jackknife_plus_ab': {'method': 'plus', 'cv': Subsample(n_resamplings=50)},
        'cv': {'method': 'base', 'cv': 10},
        'cv_plus': {'method': 'plus', 'cv': 10},
        'cv_minmax': {'method': 'minmax', 'cv': 10},
        'cqr': {'method': 'quantile', 'cv': 'split', 'alpha': alpha},
    }
    
    if isinstance(list_strategies, str): list_strategies = [list_strategies]
    if isinstance(list_strategies, (list, tuple)):
        strategies = {s: strategies_dict[s] for s in list_strategies}
    else:
        strategies = strategies_dict
    
    y_pred, y_pis = {}, {}
    uq_dict = {}
    for strategy, params in strategies.items():
        t1 = time.time()
        print(f"Running strategy >> {strategy}")
        # For quantile regressor
        if strategy == 'cqr':
            mapie = MapieQuantileRegressor(estimator_q, **params)
            mapie.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
            y_pred[strategy], ypis = mapie.predict(X_test)
        # For regressor
        else:
            mapiestrategy_file = Path(filepath, f"mapie_{strategy}.pkl")
            if not mapiestrategy_file.is_file():
                mapie = MapieRegressor(estimator, n_jobs=-1, **params)
                mapie.fit(X_train, y_train)
                pd.to_pickle(mapie, mapiestrategy_file)
            else:
                mapie = pd.read_pickle(mapiestrategy_file)
                print(f"Loaded strategy file >> {mapiestrategy_file}")
            y_pred[strategy], ypis = mapie.predict(X_test, alpha=alpha)
        y_pis[strategy] = ypis.squeeze() # Shape (N, 2 for lower and upper)
        print(f"\tTime >> {time.time() - t1}")
        sorted_idx = np.argsort(y_test)
        ylower, yupper = y_pis[strategy][:, 0], y_pis[strategy][:, 1]
        uq_dict[strategy] = {'target': np.array(y_test)[sorted_idx], 'pred': y_pred[strategy][sorted_idx], 'lower': ylower[sorted_idx], 'upper': yupper[sorted_idx],
                             'pierr_metric': {'PICP': regression_coverage_score(y_test, ylower, yupper), 'MPIW': regression_mean_width_score(ylower, yupper)}}
    
    if save_outfile: pd.to_pickle((y_pred, y_pis, uq_dict), Path(filepath, 'mapieuq_pred.pkl'))
    return y_pred, y_pis, uq_dict


def eval_pierrmetric_alpha(estimator, estimator_q, X_train, y_train, X_calib, y_calib, X_test, y_test, list_strategies=None, alphas=np.arange(0.05, 1., 0.05).round(2), filepath='./'):
    """Evaluate prediction interval coverage probability (PICP) and mean prediction interval width (MPIW) for a range of alphas. Save output files: mapieuq_picp_alpha.pkl, mapieuq_mpiw_alpha.pkl"""
    
    assert Path(filepath, 'mapie_naive.pkl').is_file(), 'Regressor estimator file not found! Run mapie_uq() once to save the file.'
    strategy_cqr = False
    if isinstance(list_strategies, (list, tuple)):
        if 'cqr' in list_strategies:
            strategy_cqr = True
            list_strategies.remove('cqr')
    
    uq_dict_alpha = {}
    for alpha in alphas:
        print(f"alpha >> {alpha}")
        uq_dict_alpha |= {alpha: {k: v['pierr_metric'] for k, v in
                                    mapie_uq(estimator, estimator_q, X_train, y_train, X_calib, y_calib, X_test, y_test,
                                             list_strategies=list_strategies, alpha=alpha, filepath=filepath, save_outfile=False)[-1].items()}}
        if strategy_cqr:
            uq_dict_alpha[alpha] |= {k: v['pierr_metric'] for k, v in
                                        mapie_uq(estimator, estimator_q, X_train, y_train, X_calib, y_calib, X_test, y_test,
                                                list_strategies='cqr', alpha=alpha, filepath=filepath, save_outfile=False)[-1].items()}
    
    # Save file
    uq_picp_alpha, uq_mpiw_alpha = {}, {}
    # dict of {strategy: scores for alphas}
    for strategy in uq_dict_alpha[list(uq_dict_alpha.keys())[0]].keys():
        uq_picp_alpha[strategy] = []
        uq_mpiw_alpha[strategy] = []
        for alpha, v in uq_dict_alpha.items():
            uq_picp_alpha[strategy].append(v[strategy]['PICP'])
            uq_mpiw_alpha[strategy].append(v[strategy]['MPIW'])
    df_picp = pd.DataFrame(uq_picp_alpha, index=uq_dict_alpha.keys())
    df_picp.to_pickle(Path(filepath, 'mapieuq_picp_alpha.pkl'))
    df_mpiw = pd.DataFrame(uq_mpiw_alpha, index=uq_dict_alpha.keys())
    df_mpiw.to_pickle(Path(filepath, 'mapieuq_mpiw_alpha.pkl'))
    
    return df_picp, df_mpiw


def main():
    # Load saved features
    datasplit_idx = pd.read_pickle(Path(logdir_exp, 'datasplitidx.pkl'))
    datasplit_idx['valid'], datasplit_idx['test'] = train_test_split(datasplit_idx['test'], test_size=cfg['frac_test_size'], random_state=cfg['seed'])
    df_features = pd.read_pickle(Path(logdir_exp, 'features.pkl'))
    yscaler = pd.read_pickle(Path(logdir_exp, 'yscaler.pkl'))
    feature_keys = [k for k in df_features.columns if isinstance(k, int)]
    dffeatures_train = df_features.loc[datasplit_idx['train']]
    dffeatures_valid = df_features.loc[datasplit_idx['valid']]
    dffeatures_test = df_features.loc[datasplit_idx['test']]
    list_strategies = ['naive', 'jackknife_plus_ab', 'cv', 'cv_plus', 'cv_minmax', 'cqr']
    
    # Run regressor
    estimator = optimize_regressor(dffeatures_train[feature_keys].to_numpy(), dffeatures_train['label'].to_numpy(), regressor=None, n_iter=100, cv=10, filename=Path(logdir_exp, 'estimator_optim.pkl'))
    estimator_q = optimize_regressor(dffeatures_train[feature_keys].to_numpy(), dffeatures_train['label'].to_numpy(), regressor='quantile', n_iter=100, cv=10, filename=Path(logdir_exp, 'estimatorq_optim.pkl')) # For quantile, if use different regressor
    
    # Run for single alpha
    y_pred, y_pis, uq_dict = mapie_uq(estimator, estimator_q, dffeatures_train[feature_keys].to_numpy(), dffeatures_train['label'].to_numpy(),
                                      dffeatures_valid[feature_keys].to_numpy(), dffeatures_valid['label'].to_numpy(),
                                      dffeatures_test[feature_keys].to_numpy(), dffeatures_test['label'].to_numpy(),
                                      list_strategies=list_strategies,
                                      alpha=0.1, filepath=logdir_exp, save_outfile=True)
    
    # Evaluate PICP and MPIW for various alphas
    df_picp, df_mpiw = eval_pierrmetric_alpha(estimator, estimator_q,
                                                   dffeatures_train[feature_keys].to_numpy(), dffeatures_train['label'].to_numpy(),
                                                   dffeatures_valid[feature_keys].to_numpy(), dffeatures_valid['label'].to_numpy(),
                                                   dffeatures_test[feature_keys].to_numpy(), dffeatures_test['label'].to_numpy(),
                                                   list_strategies=list_strategies,
                                                   alphas=np.arange(0.05, 1., 0.05).round(2), filepath=logdir_exp)


if __name__ == '__main__':
    from lib.get_config import get_config
    cfg = get_config('config/config.yaml')
    logdir_folder = '20230323/000956_LOGMBH_HB'
    # logdir_folder = '20230323/142847_LOGMBH_MGII'
    # For subsamples
    # logdir_folder = '20230309/150244_LOGMBH_HB'
    # logdir_folder = '20230309/163305_LOGMBH_MGII'
    
    logdir_exp = Path(cfg['data_dir'], 'logs', logdir_folder)
    
    main()


