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

def sum_intensity_SPICE(file):
    
    bin_facs = np.array([1, 2, 1])

    def substract_min_cube(cube):
        det_plane_min = np.nanmin(cube,axis=0)
        for i in range(0,cube.shape[0]): 
            cube[i,:,:] -= det_plane_min
        return cube

    def bindown(d,n):
        inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
        return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)
    
    #read l2 file
    exposure = read_spice_l2_fits(file,memmap=False)
    #Print all the different lines present in the file and remove the duplicata
    keys = list(exposure.keys())
    for x in keys : 
        if 'LH' in x : keys.remove(x)
        elif 'S V' and 'Extend'in x : keys.remove(x)
    print(keys)
    
    # Define parameters of source, detector, and transformation: listed in the SPICE paper
    pxsz_mu = 18
    platescale_x = pxsz_mu/1.1 # Micron per arcsecond
    platescale_y = pxsz_mu/1.1 # Micron per arcsecond
    platescale_l = pxsz_mu/0.09 # Micron per Angstrom

    #set the errors parameters
    shotnoise_fac = 0.025*np.sqrt(10)
    noisefloor = 0.07
    
    for key in keys :
        raster = exposure[key]
        if '82-' in file :
            print("crop 82")
            print('Raster shape : ',exposure[key].data.shape)
            raster = exposure[key][:,:,100:711,30:]
        else :
            print('Raster shape : ',exposure[key].data.shape)
            raster = exposure[key][:,:,100:711,:]
        
        cube = raster[0].data.transpose([2,1,0])
        cube = substract_min_cube(cube)
        raster = bindown(cube,np.round(np.array(cube.shape)/bin_facs).astype(np.int32))
        print('Size binned down : ', raster.shape)
        #mask and filter
        dat_arr = raster
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
        errors = ((noisefloor**2+np.abs(dat_filt)*shotnoise_fac**2)**0.5).astype('float32')

        sum_ampsOIII, sum_errOIII = (np.zeros([nx,ny]), ) * 2
        sum_ampsMgIX, sum_errMgIX = (np.zeros([nx,ny]), ) * 2

        sum_ampsNIV, sum_errNIV = (np.zeros([nx,ny]), ) * 2
        sum_ampsNe, sum_errNe = (np.zeros([nx,ny]), ) * 2
        sum_ampsOVI, sum_errOVI = (np.zeros([nx,ny]), ) * 2

        sum_ampsSIV750, sum_errSIV750 = (np.zeros([nx,ny]), ) * 2
        sum_ampsOIV, sum_errOIV = (np.zeros([nx,ny]), ) * 2
        sum_ampsSV, sum_errSV = (np.zeros([nx,ny]), ) * 2

        sum_ampsNa, sum_errNa = (np.zeros([nx,ny]), ) * 2
        sum_ampsNIII, sum_errNIII = (np.zeros([nx,ny]), ) * 2
    
        for i in tqdm_notebook(range(0,nx)):
            for j in range(0,ny):
                data = raster[0, :, i, j].data*u.W/(u.m**2)/u.sr/u.nm
                errs = errors[0,:,i,j]
                mask = dat_mask[0,:,i,j]
                data[mask] = 0.0*u.W/(u.m**2)/u.sr/u.nm
                if(np.sum(np.logical_not(mask)) > 5):
                    dat = data[np.logical_not(mask)].value
                    wvl = x[np.logical_not(mask)]
                
                    if 'O III 703' in key :
                        sum_ampsOIII[i][j] = np.nanmean(cube[i,j,19:38]) # O III 703
                        sum_errOIII[i][j] = np.abs((sum_ampsOIII[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                        sum_ampsMgIX[i][j] = np.nanmean(cube[i,j,38:]) # Mg 706
                        sum_errMgIX[i][j] = np.abs((sum_ampsMgIX[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    elif 'bin' in key :
                        sum_ampsSIV750[i][j] = np.nanmean(cube[i,j,12:])
                        sum_errSIV750[i][j] = np.abs((sum_ampsSIV750[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))

                    elif 'N IV 765' in key :
                        sum_ampsNIV = np.nanmean(cube[i,j,:])
                        sum_errNIV[i][j] = np.abs((sum_ampsNIV[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))

                    elif 'Ne VIII' in key :
                        sum_ampsNe = np.nanmean(cube[i,j,:])
                        sum_errNe[i][j] = np.abs((sum_ampsNe[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))


                    elif 'O VI' in key :
                        sum_ampsOVI = np.nanmean(cube[i,j,:])
                        sum_errOVI = np.abs((sum_ampsOVI[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))

                    elif 'S V 786' and 'O IV' in key :
                        sum_ampsSV[i][j] = np.nanmean(cube[i,j,:22]) 
                        sum_errSV = np.abs((sum_ampsSV[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                        sum_ampsOIV[i][j] = np.nanmean(cube[i,j,22:])
                        sum_errOIV = np.abs((sum_ampsOIV[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))

                    #N III 991 SH / LH
                    elif 'N III' in key :
                        sum_ampsNa[i][j] = np.nanmean(cube[i,j,:10])
                        sum_errNa[i][j] = np.abs((sum_ampsNa[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                        sum_ampsNIII[i][j] = np.nanmean(cube[i,j,26:])
                        sum_errNIII[i][j] = np.abs((sum_ampsNIII[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                        
    tot_sum_amps = [sum_ampsMgIX, sum_ampsOIII, sum_ampsSIV750, sum_ampsNIV, sum_ampsNe, 
                    sum_ampsSV,sum_ampsOIV,  sum_ampsNa, sum_ampsNIII, sum_ampsOVI]
        
    tot_sum_err = [sum_errMgIX, sum_errOIII, sum_errSIV750 , sum_errNIV, sum_errNe, sum_errSV, sum_errOIV, sum_errNa,   sum_errNIII,sum_errOVI ]
                        
                        
                    

    
    
    
    
    
    