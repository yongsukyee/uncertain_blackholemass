##################################################
# GET SDSS DR16 CATALOGUE AND SPECTRA
# Author: Suk Yee Yong
##################################################


from astropy.io import fits
from astropy.table import Table
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import hydra
import pandas as pd
import multiprocessing
import numpy as np
import time
import urllib


def get_catalogue(catalogue_filepath='catalogue/dr16q_prop_Oct23_2022.fits.gz', filteredcatalogue_file='sdssdr16q_prop.csv'):
    """
    Filter objects and save relevant properties to CSV
    Original SDSS DR16 Quasar Properties catalogue: http://quasar.astro.illinois.edu/paper_data/DR16Q/
    Data model: https://github.com/QiaoyaWu/sdss4_dr16q_tutorial
    """
    tab = Table(fits.getdata(catalogue_filepath, ext=1))
    
    columns = [
        'SDSS_NAME', 'PLATE', 'MJD', 'FIBERID', 'RA', 'DEC', 'OBJID',
        'Z_DR16Q', 'Z_FIT', 'Z_SYS', 'Z_SYS_ERR', 'SN_MEDIAN_ALL',
        # 'LOGL1350', 'LOGL1350_ERR', 'LOGL1700', 'LOGL1700_ERR',
        'LOGL2500', 'LOGL2500_ERR', 'LOGL3000', 'LOGL3000_ERR', 'LOGL5100', 'LOGL5100_ERR',
        'HALPHA', 'HALPHA_ERR', 'HALPHA_BR', 'HALPHA_BR_ERR',
        'HBETA', 'HBETA_ERR', 'HBETA_BR', 'HBETA_BR_ERR',
        'OIII5007', 'OIII5007_ERR', 'OIII5007C', 'OIII5007C_ERR',
        'MGII', 'MGII_ERR', 'MGII_BR', 'MGII_BR_ERR',
        # 'CIV', 'CIV_ERR',
        'LOGLBOL', 'LOGLBOL_ERR',
        'LOGMBH', 'LOGMBH_ERR', 'LOGLEDD_RATIO', 'LOGLEDD_RATIO_ERR',
        'LOGMBH_HB', 'LOGMBH_HB_ERR', 'LOGMBH_MGII', 'LOGMBH_MGII_ERR',
        # 'LOGMBH_CIV', 'LOGMBH_CIV_ERR',
    ]
    tab = tab[columns]
    # Print columns with multiple values
    # print([name for name in tab.colnames if len(tab[name].shape) > 1])
    
    list_lines = ['HALPHA', 'HBETA', 'OIII5007', 'OIII5007C', 'MGII', 'CIV']
    line_prop = ['PEAK', '50FLUX', 'FLUX', 'LOGL', 'FWHM', 'EW']
    # Unpack line measurements into separate columns
    for col in tab.colnames:
        if col.split('_')[0] in list_lines:
            for i, prop in enumerate(line_prop):
                tab[f"{col}_{prop}"] = tab[col][:,i]
            del tab[col]
    
    # Check again for multidimension columns
    # print([name for name in tab.colnames if len(tab[name].shape) > 1])
    
    # Selection criteria
    df = tab.to_pandas()
    df = df[
        (df['HBETA_FLUX']/df['HBETA_ERR_FLUX'] > 2)
        & (df['MGII_FLUX']/df['MGII_ERR_FLUX'] > 2)
        & (38 < df['HBETA_LOGL']) & (df['HBETA_LOGL'] < 48)
        & (38 < df['MGII_LOGL']) & (df['MGII_LOGL'] < 48)
        & (df['SN_MEDIAN_ALL'] >= 10) & (df['HBETA_FWHM'] != 0) & (df['MGII_FWHM'] != 0)
        & (df['LOGMBH'] != 0) & (df['LOGMBH_HB'] != 0) & (df['LOGMBH_MGII'] != 0)
        & (df['LOGMBH_ERR'] < 0.5) & (df['LOGMBH_HB_ERR'] < 0.5) & (df['LOGMBH_MGII_ERR'] < 0.5)
        & (df['HBETA_ERR_FWHM'] < 2000) & (df['HBETA_BR_ERR_FWHM'] < 2000)
        & (df['MGII_ERR_FWHM'] < 2000) & (df['MGII_BR_ERR_FWHM'] < 2000)
    ]
    print(f"Number of samples (before) >> {len(tab)}")
    print(f"Number of samples (after) >> {len(df)}")
    print(f"\tRedshift >> {df['Z_FIT'].min()} -- {df['Z_FIT'].max()}")
    print(f"\tBH mass_Hb [M_sun] >> {df['LOGMBH_HB'].min()} -- {df['LOGMBH_HB'].max()}")
    print(f"\tBH mass_MgII [M_sun] >> {df['LOGMBH_MGII'].min()} -- {df['LOGMBH_MGII'].max()}")
    df.to_csv(Path(Path(catalogue_filepath).parent, filteredcatalogue_file), index=False)
    return df


def ddl_url(url, save_filename=None, save_dir='data'):
    """Download file from URL"""
    save_filepath = Path(save_dir, save_filename)
    if save_filepath.is_file():
        print(f"\tFile existed! >> {save_filename}")
    else:
        urllib.request.urlretrieve(url, save_filepath)
        print(f"\tDownloaded file >> {save_filename}")


@hydra.main(version_base=None, config_path='config', config_name='config_getdata')
def main(cfg: DictConfig) -> None:
    # Initialize directories
    if not Path(cfg.data_dir).is_dir():
        Path(cfg.data_dir).mkdir(parents=True, exist_ok=False)
        print(f"Data folder created >> {cfg.data_dir}")
    if not Path(cfg.catalogue_dir).is_dir():
        Path(cfg.catalogue_dir).mkdir(parents=True, exist_ok=False)
        print(f"Catalogue subfolder created >> {cfg.catalogue_dir}")
    if not Path(cfg.spectra_dir).is_dir():
        Path(cfg.spectra_dir).mkdir(parents=True, exist_ok=False)
        print(f"Spectra subfolder created >> {cfg.spectra_dir}")
    if not Path(cfg.catalogue_dir, cfg.catalogue_file).is_file():
        raise FileNotFoundError(f"Catalogue file 'dr16q_prop_.fits.gz' does not exist! Download at: http://quasar.astro.illinois.edu/paper_data/DR16Q/")
    
    # --------------- #
    # Save catalogue of selected properties as CSV
    # --------------- #
    if cfg.save_filteredcatalogue:
        df = get_catalogue(catalogue_filepath=Path(cfg.catalogue_dir, cfg.catalogue_file), filteredcatalogue_file=cfg.filteredcatalogue_file)
    
    # --------------- #
    # Download spectra
    # --------------- #
    if cfg.ddl_spectra:
        df = pd.read_csv(Path(cfg.catalogue_dir, cfg.filteredcatalogue_file), sep=',', header=0)
        
        bi = 0.2
        # Find combination of cut with sufficient sample
        # import itertools
        # for mbhcut_min, mbhcut_max, bi in itertools.product([7.7,7.8,7.9,8], [8.9,9,9.1,9.2,9.3], [0.1, 0.2]):
        #     cfg.mbhcut_min, cfg.mbhcut_max = mbhcut_min, mbhcut_max
        # Sample evenly from LOGMBH cutoff
        df_mbh = df[
            (cfg.mbhcut_min<=df['LOGMBH_HB']) & (df['LOGMBH_HB'] <= cfg.mbhcut_max)
            & (cfg.mbhcut_min<=df['LOGMBH_MGII']) & (df['LOGMBH_MGII'] <= cfg.mbhcut_max)
        ].reset_index(drop=True)
        bins = np.arange(cfg.mbhcut_min, cfg.mbhcut_max+bi, bi)
        bin_labels = [f'{i:.1f}' for i in bins][1:]
        bin_mbh = pd.cut(df_mbh['LOGMBH'], bins=bins, labels=bin_labels)
        count_min = bin_mbh.value_counts().min() # Find min among the bins
        sample_idx = []
        for b in bin_labels:
            sample_idx.extend(bin_mbh[bin_mbh == b].index.to_list()[:count_min])
        df_mbh = df_mbh.iloc[sample_idx].reset_index(drop=True)
        print(f"MBH cut for bin size {bi} >> {cfg.mbhcut_min}--{cfg.mbhcut_max}")
        print(f"\tTotal sample >> {len(df_mbh)}")
        
        url = 'http://quasar.astro.illinois.edu/paper_data/DR16Q/fits/'
        print("DOWNLOADING DATA ...")
        for i in range(0, len(df_mbh), cfg.nmax_process):
            t1 = time.time()
            with multiprocessing.Pool() as pool:
                pool.starmap(ddl_url, [(f"{url}{oid.split('-')[0]}/op-{oid}.fits.gz", f"op-{oid}.fits.gz", cfg.spectra_dir) for oid in df_mbh['OBJID'][i:i+cfg.nmax_process]])
            print(f"Time elapsed >> {time.time() - t1:.4f} s")
        print(f"Saved in directory >> {cfg.spectra_dir}")
        print(f"Number of samples >> {len(df_mbh)}")


if __name__ == '__main__':
    main()

