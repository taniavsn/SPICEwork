# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:49:06 2022

@author: Tania Varesano
"""

from sunraster.instr.spice import read_spice_l2_fits
import numpy as np
import os.path
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from specutils.fitting import estimate_line_parameters
from astropy import units as u

from tqdm.notebook import tqdm_notebook
import warnings 
warnings.filterwarnings("ignore") 

from scipy.ndimage import median_filter
from astropy.nddata import StdDevUncertainty
import pickle

from specutils.fitting import fit_lines
from specutils.spectra import Spectrum1D

def sum_intensity_SPICE(filename, data_path = "."):
    data_path = input('Data path ? ')
    file = os.path.join(data_path, filename)

    #Get the spectrum of a particular line
    exposure = read_spice_l2_fits(file,memmap=False)
    print(exposure.keys())
    keys = []
    nbr_keys = int(input('Enter number of keys (max 4) : '))
    for k in range(nbr_keys) : 
        key = str(input())
        keys.append(key)
    print(keys)
    #set the errors parameters
    shotnoise_fac = 0.025*np.sqrt(10)
    noisefloor = 0.07
    tot_sum_amps = []
    tot_sum_err = []
    for key in keys :
        raster = exposure[key]
        #mask and filter
        dat_arr = raster.data
        dat_filt = median_filter(dat_arr,size=3)
        filt_thold = 1.0
        dat_median = np.nanmedian(np.abs(dat_filt))
        dat_mask = (np.isnan(dat_arr) + np.isinf(dat_arr) +
                    (np.abs(dat_arr-dat_filt) > filt_thold*(dat_median+np.abs(dat_filt)))+ (dat_arr < - 0.0)) > 0
    
        [nx,ny] = raster[0,0,:,:].data.shape
        errors = ((noisefloor**2+np.abs(dat_filt)*shotnoise_fac**2)**0.5).astype('float32')
    
        #Create an array of corresponding wavelengths for the gaussian fit
        x = raster.spectral_axis.to(u.nm)
    
        # Arrays to store the fit parameters:
        sum_amps = np.zeros([nx,ny])
        fit_err = np.zeros([nx,ny])
    
        for i in tqdm_notebook(range(0,nx)):
            for j in range(0,ny):
                data = raster[0, :, i, j].data*u.adu
                errs = errors[0,:,i,j]
                mask = dat_mask[0,:,i,j]
                data[mask] = 0.0*u.adu
                if(np.sum(np.logical_not(mask)) > 5):
                    dat = data[np.logical_not(mask)].value
                    wvl = x[np.logical_not(mask)]
                    #cont = np.min(dat)
                    sum_amps[i][j] = np.nanmean(raster[0, :, i, j].data)
                    fit_err[i][j] = np.abs((sum_amps[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))  
        tot_sum_amps.append(sum_amps)
        tot_sum_err.append(fit_err)
        
        return tot_sum_amps, tot_sum_err

