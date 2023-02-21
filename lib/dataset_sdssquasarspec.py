##################################################
# SDSS QUASAR SPECTRA DATASET
# Author: Suk Yee Yong
##################################################


from astropy.io import fits
from pathlib import Path
from sklearn import preprocessing
from torch.utils.data import Dataset

import numpy as np
import torch


class SDSSQuasarSpecDataset(Dataset):
    """
    SDSS Quasar spectra
    
    Parameters
    ----------
        listpath_fits: list of FITS file path [OBJ1, OBJ2, ...]
        df_label: DataFrame with columns ['OBJID']+label_prop
        label_key: Label key in df_label
        yscaler: sklearn.preprocessing
        transform: torchvision.transforms
    """
    def __init__(self, listpath_fits, df_label, label_key, yscaler=None, transform=None):
        self.listpath_fits = listpath_fits
        self.df_label = df_label
        self.label_key = label_key
        self.yscaler= yscaler
        self.transform = transform
    
    def __len__(self):
        return len(self.listpath_fits)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Contain: wave_prereduced, flux_prereduced, err_prereduced, wave_conti, flux_conti, err_conti, wave_line, flux_line, err_line
        specdata = fits.getdata(self.listpath_fits[idx], ext=3)
        """
        # If using flux_prereduced, need to crop to same size
        # flux = specdata['flux_prereduced']
        flux = specdata['flux_prereduced'] - specdata['flux_conti']
        # Crop to same size
        npixel = 3000
        diff_npixel = len(flux) - npixel
        slice_right = diff_npixel//2
        slice_left = diff_npixel - slice_right
        flux = flux[slice_right: -slice_left]
        """
        flux = specdata['flux_line'][:1000]
        # Scale flux [0, 1]
        scaler = preprocessing.MinMaxScaler()
        spectrum = scaler.fit_transform(flux[...,np.newaxis]).flatten()
        spectrum = torch.from_numpy(spectrum).float()
        if self.transform is not None:
            spectrum = self.transform(spectrum)
        
        objid, label = self.get_labelbyfilename(self.df_label, path_fits=self.listpath_fits[idx], objid=None)
        label = label[self.label_key].to_numpy()
        scaled_label = None
        if self.yscaler is not None:
            scaled_label = self.yscaler.transform(label[...,np.newaxis]) # Input shape (, nfeatures)
            scaled_label = torch.tensor(scaled_label).squeeze()
        label = torch.tensor(label).squeeze()
        return spectrum, label, scaled_label, objid
    
    @staticmethod
    def get_labelbyfilename(df_label, path_fits=None, objid=None):
        """Get DataFrame row associated to path_fits with filename op-<OBJID>.fits.gz from df_label or query by objid"""
        assert (path_fits is not None) or (objid is not None), f"Requires either path_fits or objid to be not None!"
        # Get OBJID from file
        if objid is None:
            if isinstance(path_fits, str):
                path_fits = [path_fits]
            # objid = [fits.getdata(pf, ext=1)['ObjID'][0] for pf in path_fits] # From FITS
            objid = [Path(pf).with_suffix('').stem.split('-',1)[-1] for pf in path_fits] # From filename
        label = df_label.loc[df_label['OBJID'].isin([objid] if isinstance(objid, str) else objid)]
        return objid, label

