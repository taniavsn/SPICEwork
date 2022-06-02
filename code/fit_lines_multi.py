# -*- coding: utf-8 -*-
"""
Created on Sun May 29 14:10:48 2022

@author: tania
"""
import os.path
import numpy as np
from sunraster.instr.spice import read_spice_l2_fits
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from specutils.fitting import estimate_line_parameters
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from scipy.ndimage import median_filter
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils import SpectralRegion

from tqdm import tqdm
import tqdm.contrib.concurrent
import pickle
import warnings 
from copy import deepcopy
warnings.filterwarnings("ignore")

from specutils.fitting import fit_lines
from specutils.spectra import Spectrum1D
from mpl_toolkits.axes_grid1 import AxesGrid
plt.rcParams['image.origin'] = 'lower'
plt.rcParams.update({'font.size': 16}) # Make the fonts in figures big enough for papers
plt.rcParams.update({'figure.figsize':[15,7]});

from astropy import constants as const 
from multiprocessing import Pool
import time
plt.rc('font', family='serif')


data_path = "C:\\Users\\tania\\EMToolKit-main\\EMToolKit-main\\mosaic"
#filename = "solo_L2_spice-n-ras_20220302T004014_V03_100663682-000.fits"
filename = "solo_L2_spice-n-ras_20220302T091034_V02_100663686-000.fits"
#filename = "solo_L2_spice-n-ras_20220302T181034_V02_100663690-000.fits"
file = os.path.join(data_path, filename)
exposure = read_spice_l2_fits(file,memmap=False)
keys  = ['O III 703 / Mg IX 706 - SH', 'S IV 750/ Mg IX (spectral bin 2)', 'N IV 765 - Peak', 'Ne VIII 770 / Mg VIII 772 - SH',
    'S V 786 / O IV 787 - LW', 'N III 991 - SH', 'O VI 1032 - Peak']


#Store the results
tot_sum_amps = []
tot_sum_errs = []
#exposure should be like exposure = read_spice_l2_fits(file, memmap=False)
#key is a str
def fit_lines_mosaic2nd(key):
    bin_facs = np.array([1, 2, 1])
    def substract_min_cube(cube):
        det_plane_min = np.nanmin(cube,axis=0)
        for i in range(0,cube.shape[0]): 
            cube[i,:,:] -= det_plane_min
        return cube

    def bindown(d,n):
        inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
        return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)    
    
    
    raster = exposure[key]
    
    #Setting constants
    pxsz_mu = 18
    platescale_x = pxsz_mu/1.1 # Micron per arcsecond
    #platescale_y = pxsz_mu/1.1 # Micron per arcsecond
    platescale_l = pxsz_mu/0.09 # Micron per Angstrom
    wl0 = (raster.meta.original_header['CRVAL3']-raster.meta.original_header['CDELT3']*raster.meta.original_header['CRPIX3'])*10
    det_scale0 = bin_facs*np.array([raster.meta.original_header['CDELT1']*platescale_x,
                                raster.meta.original_header['CDELT2']*pxsz_mu,10*raster.meta.original_header['CDELT3']*platescale_l])
    det_origin0 = np.array([0.0,0.0,wl0*platescale_l])
    
    
    wvl = raster.spectral_axis.to(u.Angstrom)
    if '82-' in file :
        print("crop 82")
        print('Raster shape : ',exposure[key].data.shape)
        raster = exposure[key][:,:,100:711,30:]
    else :
        print('Raster shape : ',exposure[key].data.shape)
        raster = exposure[key][:,:,100:711,:]
    cube = raster[0].data.transpose([2,1,0])
    cube = substract_min_cube(cube)
    
    det_dims0 = np.array(cube.shape)
    waves = (det_origin0[2]+np.arange(det_dims0[2])*det_scale0[2])/platescale_l
    wcen0 = waves[np.nanargmax(np.nansum(np.nansum(cube,axis=0),axis=0))]
    
    cube = bindown(cube,np.round(np.array(cube.shape)/bin_facs).astype(np.int32))
    print('Binned down shape : ', cube.shape)
    
    #Set error parameters
    #noisefloor based on a very low signal pixel (level of amplitude)    
    dat_arr = cube
    dat_filt = median_filter(dat_arr,size=3)
    filt_thold = 1.0
    sig0 = 0.3*u.Angstrom
    cen0 = wcen0*u.Angstrom
    dat_median = np.nanmedian(np.abs(dat_filt))
    dat_mask = (np.isnan(dat_arr) + np.isinf(dat_arr) +
                (np.abs(dat_arr-dat_filt) > filt_thold*(dat_median+np.abs(dat_filt)))+ (dat_arr < - 0.0)) > 0
    
    shotnoise_fac = 0.025*np.sqrt(10)
    noisefloor = 0.07
    errors = ((noisefloor**2+np.abs(dat_filt)*shotnoise_fac**2)**0.5).astype('float32')
    [nx,ny] = cube.shape[0:2]
 
    temp_amps = []
    temp_errs = []
    temp_dopp = []
    if key == 'O III 703 / Mg IX 706 - SH' : 
        print(key)
        fit_ampsOIII, fit_cenOIII, fit_errOIII = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        fit_ampsMgIX, fit_cenMgIX, fit_errMgIX = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        for i in tqdm.tqdm(range(0,nx)):
            for j in range(0,ny):
                data = cube[i, j, :]*u.adu      
                errs = errors[i, j, :]
                mask = dat_mask[i, j, :]
                data[mask] = 0.0*u.adu
                if(np.sum(np.logical_not(mask)) > 10):        
                    dat = data[np.logical_not(mask)].value
                    wvl =  waves[np.logical_not(mask)]*u.Angstrom
                    cont = np.min(dat)
                    wav_norm = np.trapz(dat-cont,x=wvl)
                    #amplitude, center and sigma guesses 
                    amp = np.max(dat)-cont #np.trapz(dat-cont,x=wvl)
                    cen = np.clip(np.trapz(wvl*(dat-cont),x=wvl)/wav_norm,cen0-sig0,cen0+sig0)
                    sig =np.clip((np.trapz(wvl**2*(dat-cont),x=wvl)/wav_norm-cen**2),(0.25*sig0)**2,(2.5*sig0)**2)**0.5
    
                    spec = Spectrum1D(flux = data, spectral_axis = waves*u.Angstrom, uncertainty=StdDevUncertainty(errs), mask=mask)
                    
                    sub_region = SpectralRegion(min(wvl), 702.84*u.Angstrom)
                    sub_region1 = SpectralRegion(702.84*u.Angstrom, 704.7*u.Angstrom)
                    sub_region2 = SpectralRegion(704.7*u.Angstrom, max(wvl))
                    sub_spectrum = extract_region(spec, sub_region)
                    sub_spectrum1 = extract_region(spec, sub_region1)
                    sub_spectrum2 = extract_region(spec, sub_region2)
                    g_init = estimate_line_parameters(sub_spectrum, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init1 = estimate_line_parameters(sub_spectrum1, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init2 = estimate_line_parameters(sub_spectrum2, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    c_init = models.Const1D(amplitude = noisefloor,
                                            bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + g_init1 + g_init2 + c_init)
                    y_fit = g_fit(wvl)

                    fit_ampsOIII[i][j] = g_fit.amplitude_1.value
                    fit_ampsMgIX[i][j] = g_fit.amplitude_2.value

                    fit_cenOIII[i][j] = 703.85- g_fit.mean_1.value #O III
                    fit_cenMgIX[i][j] = 706 - g_fit.mean_2.value # Mg IX

                    fit_errOIII[i][j] = np.abs((g_fit.amplitude_1.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_errMgIX[i][j] = np.abs((g_fit.amplitude_2.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
        temp_amps.append(fit_ampsOIII)
        temp_amps.append(fit_ampsMgIX)
        temp_dopp.append(fit_cenOIII)
        temp_dopp.append(fit_cenMgIX)
        temp_errs.append(fit_errOIII)
        temp_errs.append(fit_errMgIX)
        
    if key == 'S IV 750/ Mg IX (spectral bin 2)':
        print(key)
        fit_ampsSIV750, fit_cenSIV750, fit_errSIV750 = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        for i in tqdm.tqdm(range(0,nx)):
            for j in range(0,ny):
                data = cube[i, j, :]*u.adu
                errs = errors[i, j, :]
                mask = dat_mask[i, j, :]
                data[mask] = 0.0*u.adu
                if(np.sum(np.logical_not(mask)) > 10):        
                    dat = data[np.logical_not(mask)].value
                    wvl =  waves[np.logical_not(mask)]*u.Angstrom
                    cont = np.min(dat)
                    wav_norm = np.trapz(dat-cont,x=wvl)
                    #amplitude, center and sigma guesses 
                    amp = np.max(dat)-cont #np.trapz(dat-cont,x=wvl)
                    cen = np.clip(np.trapz(wvl*(dat-cont),x=wvl)/wav_norm,cen0-sig0,cen0+sig0)
                    sig =np.clip((np.trapz(wvl**2*(dat-cont),x=wvl)/wav_norm-cen**2),(0.25*sig0)**2,(2.5*sig0)**2)**0.5
    
                    spec = Spectrum1D(flux = data, spectral_axis = waves*u.Angstrom, uncertainty=StdDevUncertainty(errs), mask=mask)
                    
                    sub_region = SpectralRegion(min(wvl), 749*u.Angstrom)
                    sub_region1 = SpectralRegion(749*u.Angstrom, max(wvl))

                    sub_spectrum = extract_region(spec, sub_region)
                    sub_spectrum1 = extract_region(spec, sub_region1)
                    g_init = estimate_line_parameters(sub_spectrum, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init1 = estimate_line_parameters(sub_spectrum1, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    c_init = models.Const1D(amplitude = noisefloor, bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + g_init1 + c_init)

                    fit_ampsSIV750[i][j] = g_fit.amplitude_1.value
                    fit_cenSIV750[i][j] = 750.221 - g_fit.mean_1.value  #S IV
                    fit_errSIV750[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                    
        temp_amps.append(fit_ampsSIV750)
        temp_errs.append(fit_errSIV750)
        temp_dopp.append(fit_cenSIV750)
        
    if key == 'N IV 765 - Peak':
        print(key)
        fit_ampsNIV, fit_cenNIV, fit_errNIV = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        for i in tqdm.tqdm(range(0,nx)):
            for j in range(0,ny):
                data = cube[i, j, :]*u.adu
                errs = errors[i, j, :]
                mask = dat_mask[i, j, :]
                data[mask] = 0.0*u.adu
                if(np.sum(np.logical_not(mask)) > 10):        
                    dat = data[np.logical_not(mask)].value
                    wvl =  waves[np.logical_not(mask)]*u.Angstrom
                    cont = np.min(dat)
                    wav_norm = np.trapz(dat-cont,x=wvl)
                    #amplitude, center and sigma guesses 
                    amp = np.max(dat)-cont #np.trapz(dat-cont,x=wvl)
                    cen = np.clip(np.trapz(wvl*(dat-cont),x=wvl)/wav_norm,cen0-sig0,cen0+sig0)
                    sig =np.clip((np.trapz(wvl**2*(dat-cont),x=wvl)/wav_norm-cen**2),(0.25*sig0)**2,(2.5*sig0)**2)**0.5
    
                    spec = Spectrum1D(flux = data, spectral_axis = waves*u.Angstrom, uncertainty=StdDevUncertainty(errs), mask=mask)
                    g_init = models.Gaussian1D(amplitude=amp,
                               mean=cen,
                               stddev=sig, 
                               bounds={'amplitude':[0,np.max(dat)],
                                       'mean':[np.min(waves),np.max(waves)],
                                       'stddev':[waves[1]-waves[0],0.5*(waves[-1]-waves[0])]})
                    c_init = models.Const1D(amplitude = noisefloor,
                                                bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + c_init)
                    #y_fit = g_fit(wvl)
                    fit_ampsNIV[i][j] = g_fit.amplitude_0.value
                    fit_errNIV[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_cenNIV[i][j] = 765 - g_fit.mean_0.value
                    
                    
        temp_amps.append(fit_ampsNIV)
        temp_errs.append(fit_errNIV)
        temp_dopp.append(fit_cenNIV)
    
    if key == 'Ne VIII 770 / Mg VIII 772 - SH':
        print(key)
        fit_ampsNe, fit_cenNe, fit_errNe = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        for i in tqdm.tqdm(range(0,nx)):
            for j in range(0,ny):
                data = cube[i, j, :]*u.adu
                errs = errors[i, j, :]
                mask = dat_mask[i, j, :]
                data[mask] = 0.0*u.adu
                if(np.sum(np.logical_not(mask)) > 10):        
                    dat = data[np.logical_not(mask)].value
                    wvl =  waves[np.logical_not(mask)]*u.Angstrom
                    cont = np.min(dat)
                    wav_norm = np.trapz(dat-cont,x=wvl)
                    #amplitude, center and sigma guesses 
                    amp = np.max(dat)-cont #np.trapz(dat-cont,x=wvl)
                    cen = np.clip(np.trapz(wvl*(dat-cont),x=wvl)/wav_norm,cen0-sig0,cen0+sig0)
                    sig =np.clip((np.trapz(wvl**2*(dat-cont),x=wvl)/wav_norm-cen**2),(0.25*sig0)**2,(2.5*sig0)**2)**0.5
    
                    spec = Spectrum1D(flux = data, spectral_axis = waves*u.Angstrom, uncertainty=StdDevUncertainty(errs), mask=mask)
                    g_init = models.Gaussian1D(amplitude=amp,
                               mean=cen,
                               stddev=sig, 
                               bounds={'amplitude':[0,np.max(dat)],
                                       'mean':[np.min(waves),np.max(waves)],
                                       'stddev':[waves[1]-waves[0],0.5*(waves[-1]-waves[0])]})
                    c_init = models.Const1D(amplitude = noisefloor,
                                                bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + c_init)
                    fit_ampsNe[i][j] = g_fit.amplitude_0.value
                    fit_errNe[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_cenNe[i][j] = 770 - g_fit.mean_0.value
                    
        temp_amps.append(fit_ampsNe)
        temp_errs.append(fit_errNe)
        temp_dopp.append(fit_cenNe)
        
    if key == 'S V 786 / O IV 787 - LW' :
        print(key)
        fit_ampsOIV, fit_cenOIV, fit_errOIV = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        fit_ampsSV, fit_cenSV, fit_errSV = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        for i in tqdm.tqdm(range(0,nx)):
            for j in range(0,ny):
                data = cube[i, j, :]*u.adu
                errs = errors[i, j, :]
                mask = dat_mask[i, j, :]
                data[mask] = 0.0*u.adu
                if(np.sum(np.logical_not(mask)) > 10):        
                    dat = data[np.logical_not(mask)].value
                    wvl =  waves[np.logical_not(mask)]*u.Angstrom
                    cont = np.min(dat)
                    wav_norm = np.trapz(dat-cont,x=wvl)
                    #amplitude, center and sigma guesses 
                    amp = np.max(dat)-cont #np.trapz(dat-cont,x=wvl)
                    cen = np.clip(np.trapz(wvl*(dat-cont),x=wvl)/wav_norm,cen0-sig0,cen0+sig0)
                    sig =np.clip((np.trapz(wvl**2*(dat-cont),x=wvl)/wav_norm-cen**2),(0.25*sig0)**2,(2.5*sig0)**2)**0.5
    
                    spec = Spectrum1D(flux = data, spectral_axis = waves*u.Angstrom, uncertainty=StdDevUncertainty(errs), mask=mask)
                    sub_region = SpectralRegion(min(wvl), 787*u.Angstrom)
                    sub_region1 = SpectralRegion(787*u.Angstrom, max(wvl))

                    sub_spectrum = extract_region(spec, sub_region)
                    sub_spectrum1 = extract_region(spec, sub_region1)

                    g_init = estimate_line_parameters(sub_spectrum, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init1 = estimate_line_parameters(sub_spectrum1, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    c_init = models.Const1D(amplitude = noisefloor, bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + g_init1 + c_init)
                    #y_fit = g_fit(wvl)                           
                    fit_ampsSV[i][j] = g_fit.amplitude_0.value
                    fit_ampsOIV[i][j] = g_fit.amplitude_1.value

                    fit_cenSV[i][j] = 786.47 - g_fit.mean_0.value  # S V 786
                    fit_cenOIV[i][j] = 787.71 - g_fit.mean_1.value #O IV 787

                    fit_errSV[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_errOIV[i][j] = np.abs((g_fit.amplitude_1.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
        temp_amps.append(fit_ampsOIV)
        temp_errs.append(fit_errOIV)
        temp_dopp.append(fit_cenOIV)
        temp_amps.append(fit_ampsSV)
        temp_errs.append(fit_errSV)
        temp_dopp.append(fit_cenSV)
        
    if key == 'N III 991 - SH' :
        print(key)
        fit_ampsNa, fit_cenNa, fit_errNa = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        fit_ampsNIII, fit_cenNIII, fit_errNIII = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        for i in tqdm.tqdm(range(0,nx)):
            for j in range(0,ny):
                data = cube[i, j, :]*u.adu
                errs = errors[i, j, :]
                mask = dat_mask[i, j, :]
                data[mask] = 0.0*u.adu
                if(np.sum(np.logical_not(mask)) > 10):        
                    dat = data[np.logical_not(mask)].value
                    wvl =  waves[np.logical_not(mask)]*u.Angstrom
                    cont = np.min(dat)
                    wav_norm = np.trapz(dat-cont,x=wvl)
                    #amplitude, center and sigma guesses 
                    amp = np.max(dat)-cont #np.trapz(dat-cont,x=wvl)
                    cen = np.clip(np.trapz(wvl*(dat-cont),x=wvl)/wav_norm,cen0-sig0,cen0+sig0)
                    sig =np.clip((np.trapz(wvl**2*(dat-cont),x=wvl)/wav_norm-cen**2),(0.25*sig0)**2,(2.5*sig0)**2)**0.5
    
                    spec = Spectrum1D(flux = data, spectral_axis = waves*u.Angstrom, uncertainty=StdDevUncertainty(errs), mask=mask) 
                    sub_region = SpectralRegion(min(wvl), 989*u.Angstrom)
                    sub_region1 = SpectralRegion(989*u.Angstrom, 990.6*u.Angstrom)
                    sub_region2 = SpectralRegion(990.6*u.Angstrom, max(wvl))
                    sub_spectrum = extract_region(spec, sub_region)
                    sub_spectrum1 = extract_region(spec, sub_region1)
                    sub_spectrum2 = extract_region(spec, sub_region2)
                    g_init = estimate_line_parameters(sub_spectrum, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init1 = estimate_line_parameters(sub_spectrum1, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init2 = estimate_line_parameters(sub_spectrum2, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    c_init = models.Const1D(amplitude = noisefloor,
                                            bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + g_init1 + g_init2 + c_init)

                    fit_ampsNa[i][j] = g_fit.amplitude_0.value
                    fit_ampsNIII[i][j] = g_fit.amplitude_2.value
                    fit_cenNa[i][j] = 988.6 - g_fit.mean_0.value  #Na VI 2s2 2p2 3P2 - 2s 2p3 5S2 
                    fit_cenNIII[i][j] = 991.51 - g_fit.mean_2.value # N III 2s2 2p 2P3/2 - 2s 2p2 2 D3-5/2 (2 lines) 
                    fit_errNa[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_errNIII[i][j] = np.abs((g_fit.amplitude_2.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                    
        temp_amps.append(fit_ampsNa)
        temp_errs.append(fit_errNa)
        temp_dopp.append(fit_cenNa)
        temp_amps.append(fit_ampsNIII)
        temp_errs.append(fit_errNIII)
        temp_dopp.append(fit_cenNIII)
        
    if key == 'O VI 1032 - Peak' :
        print(key)
        fit_ampsOVI, fit_cenOVI, fit_errOVI = np.zeros([nx,ny]), np.zeros([nx,ny]), np.zeros([nx,ny])
        for i in tqdm.tqdm(range(0,nx)):
            for j in range(0,ny):
                data = cube[i, j, :]*u.adu
                errs = errors[i, j, :]
                mask = dat_mask[i, j, :]
                data[mask] = 0.0*u.adu
                if(np.sum(np.logical_not(mask)) > 10):        
                    dat = data[np.logical_not(mask)].value
                    wvl =  waves[np.logical_not(mask)]*u.Angstrom
                    cont = np.min(dat)
                    wav_norm = np.trapz(dat-cont,x=wvl)
                    #amplitude, center and sigma guesses 
                    amp = np.max(dat)-cont #np.trapz(dat-cont,x=wvl)
                    cen = np.clip(np.trapz(wvl*(dat-cont),x=wvl)/wav_norm,cen0-sig0,cen0+sig0)
                    sig =np.clip((np.trapz(wvl**2*(dat-cont),x=wvl)/wav_norm-cen**2),(0.25*sig0)**2,(2.5*sig0)**2)**0.5
    
                    spec = Spectrum1D(flux = data, spectral_axis = waves*u.Angstrom, uncertainty=StdDevUncertainty(errs), mask=mask)
                    g_init = models.Gaussian1D(amplitude=amp,
                               mean=cen,
                               stddev=sig, 
                               bounds={'amplitude':[0,np.max(dat)],
                                       'mean':[np.min(waves),np.max(waves)],
                                       'stddev':[waves[1]-waves[0],0.5*(waves[-1]-waves[0])]})
                    c_init = models.Const1D(amplitude = noisefloor,
                                                bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + c_init)
                    #y_fit = g_fit(wvl)
                    fit_ampsOVI[i][j] = g_fit.amplitude_0.value
                    fit_errOVI[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_cenOVI[i][j] = 1032 - g_fit.mean_0.value
                    
        temp_amps.append(fit_ampsOVI)
        temp_errs.append(fit_errOVI)
        temp_dopp.append(fit_cenOVI)
        
    return temp_amps, temp_errs, temp_dopp
    
keys = ['O III 703 / Mg IX 706 - SH', 'S IV 750/ Mg IX (spectral bin 2)', 'N IV 765 - Peak', 'Ne VIII 770 / Mg VIII 772 - SH',
         'S V 786 / O IV 787 - LW', 'N III 991 - SH', 'O VI 1032 - Peak']

if __name__ == "__main__":
    res = tqdm.contrib.concurrent.process_map(fit_lines_mosaic2nd, keys)
   
    totfitfile = 'total_fit_86_adu.json'
    with open(totfitfile, 'wb') as famps:
        pickle.dump(res, famps)  