# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:13:16 2022

@author: Tania
"""

import os.path
import numpy as np
from sunraster.instr.spice import read_spice_l2_fits
import matplotlib.pyplot as plt
#import fiplcr # Natalia's package

from astropy.modeling import models, fitting
from specutils.fitting import estimate_line_parameters
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from scipy.ndimage import median_filter
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils import SpectralRegion

from tqdm.notebook import tqdm_notebook
import tqdm.contrib.concurrent
import pickle
import warnings 
from copy import deepcopy
warnings.filterwarnings("ignore")

from specutils.fitting import fit_lines
from specutils.spectra import Spectrum1D
# import EMToolKit.instruments.spice_functions_abundance as sfab
# from EMToolKit.instruments.spice import contribution_func_spice
from mpl_toolkits.axes_grid1 import AxesGrid
plt.rcParams['image.origin'] = 'lower'
plt.rcParams.update({'font.size': 16}) # Make the fonts in figures big enough for papers
plt.rcParams.update({'figure.figsize':[15,7]});

from astropy import constants as const 
from multiprocessing import Pool
import time
import matplotlib as mlt
mlt.rc('xtick', labelsize=18)
mlt.rc('ytick', labelsize=18)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import matplotlib.colors as colors

data_path = "mosaic"
filename = "solo_L2_spice-n-ras_20220302T181034_V02_100663690-000.fits"
file = os.path.join(data_path, filename)
exposure = read_spice_l2_fits(file,memmap=False)


#Downsize the image for reducing the complexity
#parameters are d : datacube (shape X Y lambda), n the new dimensions of the cube (array([X, Y, lambda]))
def bindown(d,n, bin_facs=np.array([1, 2, 1])):
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)


def substract_min_cube(cube):
    det_plane_min = np.nanmin(cube,axis=0)
    for i in range(0,cube.shape[0]): 
        cube[i,:,:] -= det_plane_min
    return cube

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
bin_facs = np.array([1, 2, 1]) #bin down by a factor 2 on the Y axis

def fit_lines_mosaic2nd(key):    
    #crop the dumbbells
    if '82-' in filename :
        print("crop 82")
        print('Raster shape : ',exposure[key].data.shape)
        raster = exposure[key][:,:,100:711,30:]
    else :
        print('Raster shape : ',exposure[key].data.shape)
        raster = exposure[key][:,:,100:711,:]
    
    cube = raster[0].data.transpose([2,1,0])
    cube = substract_min_cube(cube)
    print('Shape before binning : ',cube.shape)
    wl0 = (raster.meta.original_header['CRVAL3']-raster.meta.original_header['CDELT3']*raster.meta.original_header['CRPIX3'])*10
    det_scale0 = bin_facs*np.array([raster.meta.original_header['CDELT1']*platescale_x,
                                raster.meta.original_header['CDELT2']*pxsz_mu,10*raster.meta.original_header['CDELT3']*platescale_l])
    det_origin0 = np.array([0.0,0.0,wl0*platescale_l])
    det_dims0 = np.array(cube.shape)
    waves = (det_origin0[2]+np.arange(det_dims0[2])*det_scale0[2])/platescale_l
    wcen0 = waves[np.nanargmax(np.nansum(np.nansum(cube,axis=0),axis=0))]
    
    #reduce the size of the raster
    raster = bindown(cube,np.round(np.array(cube.shape)/bin_facs).astype(np.int32))
    print('Size binned down : ', raster.shape)
    #mask and filter
    dat_arr = raster
    dat_filt = median_filter(dat_arr,size=3)
    filt_thold = 0.5
    sig0 = 0.3*u.Angstrom
    cen0 = wcen0*u.Angstrom
    dat_median = np.nanmedian(np.abs(dat_filt))
    dat_mask = (np.isnan(dat_arr) + np.isinf(dat_arr) +
                (np.abs(dat_arr-dat_filt) > filt_thold*(dat_median+np.abs(dat_filt)))+ (dat_arr < - 0.0)) > 0
    
    #Initialization of arrays for storing fit parameters
    [nx,ny] = raster.shape[0:2]
    errors = ((noisefloor**2+np.abs(dat_filt)*shotnoise_fac**2)**0.5).astype('float32')
    
    fit_ampsOIII, fit_cenOIII, fit_sigsOIII, fit_errOIII = (np.zeros([nx,ny]), ) * 4
    fit_ampsMgIX, fit_cenMgIX, fit_sigsMgIX, fit_errMgIX = (np.zeros([nx,ny]), ) * 4
    
    fit_ampsSIV750, fit_cenSIV750, fit_sigsSIV750, fit_errSIV750 = (np.zeros([nx,ny]), ) * 4
    
    fit_ampsNIV, fit_cenNIV, fit_sigsNIV, fit_errNIV = (np.zeros([nx,ny]), ) * 4
    fit_ampsNe, fit_cenNe, fit_sigsNe, fit_errNe = (np.zeros([nx,ny]), ) * 4
    fit_ampsOVI, fit_cenOVI, fit_sigsOVI, fit_errOVI = (np.zeros([nx,ny]), ) * 4
    
    fit_ampsSIV750, fit_cenSIV750, fit_sigsSIV750, fit_errSIV750 = (np.zeros([nx,ny]), ) * 4
    fit_ampsSIV748, fit_cenSIV748, fit_sigsSIV748, fit_errSIV748 = (np.zeros([nx,ny]), ) * 4
    fit_ampsOIV, fit_cenOIV, fit_sigsOIV, fit_errOIV = (np.zeros([nx,ny]), ) * 4
    fit_ampsSV, fit_cenSV, fit_sigsSV, fit_errSV = (np.zeros([nx,ny]), ) * 4
    
    fit_ampsNa, fit_cenNa, fit_sigsNa, fit_errNa = (np.zeros([nx,ny]), ) * 4
    fit_ampsNIII, fit_cenNIII, fit_sigsNIII, fit_errNIII = (np.zeros([nx,ny]), ) * 4
    
    #Run for each pixel of the raster
    for i in tqdm_notebook(range(0,nx)):
        for j in range(0,ny):
            data = raster[i, j, :]*u.W/(u.m**2)/u.sr/u.nm
            errs = errors[i, j, :]
            mask = dat_mask[i, j, :]

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
                
                
                
                if 'O III 703' in key :
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

                    fit_sigsOIII[i][j] = g_fit.stddev_1.value
                    fit_sigsMgIX[i][j] = g_fit.stddev_2.value

                    fit_cenOIII[i][j] = 703.85- g_fit.mean_1.value #O III
                    fit_cenMgIX[i][j] = 706 - g_fit.mean_2.value # Mg IX

                    fit_errOIII[i][j] = np.abs((g_fit.amplitude_1.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_errMgIX[i][j] = np.abs((g_fit.amplitude_2.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                elif 'bin' in key :
                    sub_region = SpectralRegion(min(wvl), 749*u.Angstrom)
                    sub_region1 = SpectralRegion(749*u.Angstrom, max(wvl))

                    sub_spectrum = extract_region(spec, sub_region)
                    sub_spectrum1 = extract_region(spec, sub_region1)
                    g_init = estimate_line_parameters(sub_spectrum, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init1 = estimate_line_parameters(sub_spectrum1, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    c_init = models.Const1D(amplitude = noisefloor, bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + g_init1 + c_init)
                    y_fit = g_fit(wvl)

                    fit_ampsSIV750[i][j] = g_fit.amplitude_1.value
                    fit_sigsSIV750[i][j] = g_fit.stddev_1.value
                    fit_cenSIV750[i][j] = 750.221 - g_fit.mean_1.value  #S IV

                    fit_errSIV750[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))

                elif 'N IV 765' in key :
                    g_init = models.Gaussian1D(amplitude=amp,
                               mean=cen,
                               stddev=sig, 
                               bounds={'amplitude':[0,np.max(dat)],
                                       'mean':[np.min(waves),np.max(waves)],
                                       'stddev':[waves[1]-waves[0],0.5*(waves[-1]-waves[0])]})
                    c_init = models.Const1D(amplitude = noisefloor,
                                                bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + c_init)
                    y_fit = g_fit(wvl)
                    fit_ampsNIV[i][j] = g_fit.amplitude_0.value
                    fit_sigsNIV[i][j] = g_fit.stddev_0.value
                    fit_errNIV[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_cenNIV[i][j] = 765 - g_fit.mean_0.value
                
                elif 'Ne VIII' in key :
                    g_init = models.Gaussian1D(amplitude=amp,
                               mean=cen,
                               stddev=sig, 
                               bounds={'amplitude':[0,np.max(dat)],
                                       'mean':[np.min(waves),np.max(waves)],
                                       'stddev':[waves[1]-waves[0],0.5*(waves[-1]-waves[0])]})
                    c_init = models.Const1D(amplitude = noisefloor,
                                                bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + c_init)
                    y_fit = g_fit(wvl)
                    fit_ampsNe[i][j] = g_fit.amplitude_0.value
                    fit_sigsNe[i][j] = g_fit.stddev_0.value
                    fit_errNe[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_cenNe[i][j] = 770 - g_fit.mean_0.value
                
                
                elif 'O VI' in key :
                    g_init = models.Gaussian1D(amplitude=amp,
                               mean=cen,
                               stddev=sig, 
                               bounds={'amplitude':[0,np.max(dat)],
                                       'mean':[np.min(waves),np.max(waves)],
                                       'stddev':[waves[1]-waves[0],0.5*(waves[-1]-waves[0])]})
                    c_init = models.Const1D(amplitude = noisefloor,
                                                bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + c_init)
                    y_fit = g_fit(wvl)
                    fit_ampsOVI[i][j] = g_fit.amplitude_0.value
                    fit_sigsOVI[i][j] = g_fit.stddev_0.value
                    fit_errOVI[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_cenOVI[i][j] = 1032 - g_fit.mean_0.value
                
                elif 'S V 786' and 'O IV' in key :
                    sub_region = SpectralRegion(min(wvl), 787*u.Angstrom)
                    sub_region1 = SpectralRegion(787*u.Angstrom, max(wvl))

                    sub_spectrum = extract_region(spec, sub_region)
                    sub_spectrum1 = extract_region(spec, sub_region1)

                    g_init = estimate_line_parameters(sub_spectrum, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init1 = estimate_line_parameters(sub_spectrum1, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    c_init = models.Const1D(amplitude = noisefloor, bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + g_init1 + c_init)
                    y_fit = g_fit(wvl)                           
                    fit_ampsSV[i][j] = g_fit.amplitude_0.value
                    fit_ampsOIV[i][j] = g_fit.amplitude_1.value

                    fit_sigsSV[i][j] = g_fit.stddev_0.value
                    fit_sigsOIV[i][j] = g_fit.stddev_1.value

                    fit_cenSV[i][j] = 786.47 - g_fit.mean_0.value  # S V 786
                    fit_cenOIV[i][j] = 787.71 - g_fit.mean_1.value #O IV 787

                    fit_errSV[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_errOIV[i][j] = np.abs((g_fit.amplitude_1.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))

                #S IV 748 - Extended
                elif 'S IV' and 'Extended' in key :
                    sub_region = SpectralRegion(min(wvl), 749.1*u.Angstrom)
                    sub_region1 = SpectralRegion(749.1*u.Angstrom, max(wvl))

                    sub_spectrum = extract_region(spec, sub_region)
                    sub_spectrum1 = extract_region(spec, sub_region1)

                    g_init = estimate_line_parameters(sub_spectrum, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    g_init1 = estimate_line_parameters(sub_spectrum1, models.Gaussian1D(bounds = {'stddev':[0,3]}))
                    c_init = models.Const1D(amplitude = noisefloor, bounds={'amplitude':[np.min(dat),np.max(dat)]})

                    g_fit = fit_lines(spec, g_init + g_init1 + c_init)
                    y_fit = g_fit(wvl)                           
                    fit_ampsSIV748[i][j] = g_fit.amplitude_1.value

                    fit_sigsSIV748[i][j] = g_fit.stddev_1.value

                    fit_cenSIV748[i][j] = 750.2 - g_fit.mean_1.value # S IV intensity 10e3

                    fit_errSIV748[i][j] = np.abs((g_fit.amplitude_1.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                
                #N III 991 SH / LH
                elif 'N III' in key :
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

                    fit_sigsNa[i][j] = g_fit.stddev_0.value
                    fit_sigsNIII[i][j] = g_fit.stddev_2.value

                    fit_cenNa[i][j] = 988.6 - g_fit.mean_0.value  #Na VI 2s2 2p2 3P2 - 2s 2p3 5S2 
                    fit_cenNIII[i][j] = 991.51 - g_fit.mean_2.value # N III 2s2 2p 2P3/2 - 2s 2p2 2 D3-5/2 (2 lines) 

                    fit_errNa[i][j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                    fit_errNIII[i][j] = np.abs((g_fit.amplitude_2.value/np.nansum(data[mask==0].value))*
                                           np.sqrt(np.nansum((errs[mask==0])**2)))
                        
    tot_fit_amps = [ fit_ampsOIII, fit_ampsMgIX, fit_ampsSIV750, fit_ampsNIV, fit_ampsNe, fit_ampsSV,
                    fit_ampsOIV, fit_ampsSIV748, fit_ampsNa, fit_ampsNIII, fit_ampsOVI]
    tot_fit_errors = [fit_errOIII, fit_errMgIX, fit_errSIV750, fit_errNIV, fit_errNe, fit_errSV,
                      fit_errOIV, fit_errSIV748, fit_errNa, fit_errNIII, fit_errOVI]
    tot_fit_shifts = [fit_cenOIII, fit_cenMgIX, fit_cenSIV750, fit_cenNIV, fit_cenNe, fit_cenSV,
                    fit_cenOIV, fit_cenSIV748, fit_cenNa, fit_cenNIII, fit_cenOVI]
    
    file_amps = 'all_amps' + filename + '.json'
    file_errs = 'all_errs' + filename + '.json'
    file_dopp = 'all_shifts' + filename + '.json'
    with open(file_amps, 'wb') as famps:
        pickle.dump(tot_fit_amps, famps)
    with open(file_errs, 'wb') as ferr:
        pickle.dump(tot_fit_errors, ferr)
    with open(file_dopp, 'wb') as fshift:
        pickle.dump(tot_fit_shifts, fshift)
    return tot_fit_amps, tot_fit_errors, tot_fit_shifts

keys = ['O III 703 / Mg IX 706 - SH', 'S IV 750/ Mg IX (spectral bin 2)', 'N IV 765 - Peak', 'Ne VIII 770 / Mg VIII 772 - SH',
        'S V 786 / O IV 787 - Extend','S V 786 / O IV 787 - LW', 'N III 991 - SH', 'O VI 1032 - Peak']

if __name__ == "__main__":
    tot_fit_amps, tot_fit_errors, tot_fit_shifts = tqdm.contrib.concurrent.process_map(fit_lines_mosaic2nd, keys)
