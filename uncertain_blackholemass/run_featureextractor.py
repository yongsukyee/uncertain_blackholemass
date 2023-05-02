##################################################
# PREDICT BLACK HOLE MASS FROM SDSS QUASAR SPECTRA
# Author: Suk Yee Yong
##################################################


from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import hydra
import logging
import numpy as np
import pandas as pd
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from lib.dataset_sdssquasarspec import SDSSQuasarSpecDataset
from models.encoder import DenseEncoder, ConvEncoder


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.seed is None:
        cfg.seed = int(time.time())
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Get log path from Hydra
    logdir_exp = Path(hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'])
    t1 = time.time()
    
    # --------------- #
    # Prepare dataset
    # --------------- #
    # List files
    list_files = np.asarray([str(f) for f in Path(cfg.spectra_dir).rglob('*.fits.gz')])
    # Labels
    df_catalogue = pd.read_csv(Path(cfg.catalogue_dir, cfg.filteredcatalogue_file), sep=',', header=0) # All objects
    df_sample = SDSSQuasarSpecDataset.get_labelbyfilename(df_catalogue, path_fits=list_files)[1] # DataFrame for sample
    
    # Scale label
    y_train = df_sample[cfg.logmbh_line]
    yscaler = preprocessing.MinMaxScaler().fit(np.asarray(y_train)[...,np.newaxis])
    pd.to_pickle(yscaler, Path(logdir_exp, 'yscaler.pkl'))
    
    # Split train, valid, test sets
    nsamples = len(list_files)
    train_idx, test_idx = train_test_split(np.arange(nsamples), test_size=cfg.frac_test_size, random_state=cfg.seed)
    # Save index to file
    pd.to_pickle({'train': train_idx, 'test': test_idx}, Path(logdir_exp, 'datasplitidx.pkl'))
    
    # Dataset and DataLoader
    train_dataset = SDSSQuasarSpecDataset(list_files[train_idx], df_label=df_catalogue, label_key=cfg.logmbh_line, yscaler=yscaler)
    test_dataset = SDSSQuasarSpecDataset(list_files[test_idx], df_label=df_catalogue, label_key=cfg.logmbh_line, yscaler=yscaler)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    log.info(f"NUMBER OF SAMPLES ...")
    log.info(f"\tTotal >> {nsamples}")
    log.info(f"\tTrain >> {len(train_dataset)}")
    log.info(f"\tTest >> {len(test_dataset)}")
    
    # --------------- #
    # Model
    # --------------- #
    model_kwargs = {'input_shape': train_dataset[0][0].shape, 'num_labels': 1, 'list_linear': [64, 64, 8], 'dropout': 0.1}
    model = DenseEncoder(**model_kwargs).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=0.5, total_iters=2)
    best_loss = float('inf')
    train_losses, test_losses = [], []
    
    # Add hook for feature extraction
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # --------------- #
    # Run model
    # --------------- #
    for epoch in range(cfg.max_epochs):
        features, outputs = [], []
        labels, scaled_labels, objids = [], [], []
        train_loss = 0.
        test_loss = 0.
        # For feature extraction
        extract_layer = model.net[-5]
        extract_layername = type(extract_layer).__name__ + str(extract_layer.out_features)
        fe_hook = extract_layer.register_forward_hook(get_activation(extract_layername))
        
        # Train model
        model.train()
        for (spectrum, label, scaled_label, objid) in train_dataloader:
            spectrum, scaled_label = spectrum.to(device), scaled_label.to(device)
            optimizer.zero_grad()
            output = model(spectrum)
            objids.extend(*objid) # objid is tuple
            outputs.extend(output.detach().cpu().squeeze().numpy())
            labels.extend(label.detach().cpu().squeeze().numpy())
            scaled_labels.extend(scaled_label.detach().cpu().squeeze().numpy())
            features.extend(activation[extract_layername].detach().cpu().squeeze().numpy())
            loss = criterion(output, scaled_label.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()*spectrum.size(0)
        
        # Validation
        model.eval()
        with torch.inference_mode():
            for (spectrum, label, scaled_label, objid) in test_dataloader:
                spectrum, scaled_label = spectrum.to(device), scaled_label.to(device)
                output = model(spectrum)
                objids.extend(*objid)
                outputs.extend(output.detach().cpu().squeeze().numpy())
                labels.extend(label.detach().cpu().squeeze().numpy())
                scaled_labels.extend(scaled_label.detach().cpu().squeeze().numpy())
                features.extend(activation[extract_layername].detach().cpu().squeeze().numpy())
                loss = criterion(output, scaled_label.unsqueeze(1).float())
                test_loss += loss.item()*spectrum.size(0)
        fe_hook.remove()
        
        # Calculate average losses
        train_loss /= len(train_dataloader.sampler)
        train_losses.append(train_loss)
        test_loss /= len(test_dataloader.sampler)
        test_losses.append(test_loss)
        log.info(f"epoch: [{epoch}/{cfg.max_epochs}] | train loss: {train_loss:.5g} | test loss: {test_loss:.5g}")
        
        # Track and save best model
        if test_loss < best_loss:
            log.info(f"\tTest loss decreased >> {best_loss:.5g} -> {test_loss:.5g}")
            log.info(f"\tSaved model >> {Path(logdir_exp, 'model.pth')}")
            torch.save(model.state_dict(), Path(logdir_exp, 'model.pth'))
            df_features = pd.concat([pd.DataFrame(features), pd.DataFrame({'objid': objids, 'label': labels, 'scaled_label': scaled_labels, 'output': outputs})], axis=1)
            df_features.sort_values('objid', ascending=True, ignore_index=True).to_pickle(Path(logdir_exp, 'features.pkl'))
            best_loss = test_loss
    pd.DataFrame({'train': train_losses, 'test': test_losses}).to_pickle(Path(logdir_exp, 'loss.pkl'))
    log.info(f"Time >> {time.time() - t1:g}")


if __name__ == '__main__':
    main()

