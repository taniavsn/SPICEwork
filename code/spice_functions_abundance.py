# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:13:41 2022

@author: Tania Varesano
"""
from sunraster.instr.spice import read_spice_l2_fits
import numpy as np
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from astropy.modeling import models, fitting
from specutils.fitting import estimate_line_parameters
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from scipy.ndimage import median_filter

from tqdm.notebook import tqdm_notebook
import pickle
import warnings 
warnings.filterwarnings("ignore")

from specutils.fitting import fit_lines
from specutils.spectra import Spectrum1D

from astropy.visualization import quantity_support
quantity_support()
from fiasco import Ion

from EMToolKit.instruments.spice import contribution_func_spice
from EMToolKit.algorithms.simple_reg_dem_wrapper import simple_reg_dem_wrapper
import EMToolKit.EMToolKit_SPICE as emtk

#Extract the ions and wavelength from the chosen keys
def extract_ions_wvl(keys):
    wvl = []
    ions = []
    for i in keys:
        j = i.split(' ')[0] + ' ' + i.split(' ')[1]
        wvlen = i.split(' ')[2]
        if (i.split(' ')[0] == 'Ly'):
            j = 'H I'
            wvlen = 1025
            warnings.warn('Line Ly Beta has been set to ion H I')
            wvl.append(int(wvlen))
            ions.append(j)
        elif (i.split(' ')[0] == 'Ly-gamma-CIII'):
            j = 'C III'
            wvlen = 977
            warnings.warn('Line Ly Gamma - CIII has been set to ion C III')
            wvl.append(int(wvlen))
            ions.append(j)
        elif (i.split(' ')[0] == '(STP122)' or i.split(' ')[0] == '(Ref)'):
            j = i.split(' ')[1] + ' ' + i.split(' ')[2]
            wvlen = i.split(' ')[3] 
            wvl.append(int(wvlen))
            ions.append(j)
        else : 
            wvl.append(int(wvlen))
            ions.append(j)
    return ions, wvl 


#Plot the colorbar : coronal abundance => colors, intensity => dark/light
def clrbar_ab_int(dimx=256, dimy=32):
    cb_img = np.ones([dimx,dimy,3])
    for i in range(0,dimx): # Set colors in array:
        cb_img[i,:,0] = (dimx-1-i)/(dimx-1)   # Red channel
        cb_img[i,:,2] = 1-cb_img[i,:,0]     # Blue channel
    for i in range(0,dimy): # Set intensities in array:
        cb_img[:,i,:] *= i/(dimy-1)    
    return cb_img

#calculate the coronal abundance for which the chi2 is minimum (best guess)
def chi2_mins(list_chi2):
    chi2s = np.dstack(list_chi2)
    chi_mins = np.zeros([chi2s.shape[0],chi2s.shape[1]])
    chi_mins_idx = np.zeros([chi2s.shape[0],chi2s.shape[1]])
    for i in range(chi2s.shape[0]):
        for j in range(chi2s.shape[1]):
            chi_mins[i,j] = min(list(chi2s[i,j]))
            chi_mins_idx[i,j] = list(chi2s[i,j]).index(min(list(chi2s[i,j])))
    ##PLot the chi2 min values
    plt.figure(constrained_layout=True)
    plt.imshow((chi_mins/chi_mins[200:630,:].max()),
                   vmax= 1, extent=[0,160*2.5,125,725])   
    clrbr = plt.colorbar()
    clrbr.set_label('Min chi2 value')
    plt.show()

    return chi_mins, chi_mins_idx

def plot_abundances_intensity(keys, tot_fit_amps, chi_mins, chi_mins_idx, cropx=125, cropy=725):   
    cb_img = clrbar_ab_int()
    plt.figure(figsize=[15,15])
    gfac=1/2.2
    clrimg = np.zeros([chi_mins.shape[0], chi_mins.shape[1], 3])
    gimg = np.ones(chi_mins_idx.shape)
    bimg = chi_mins_idx/20
    rimg = np.ones(chi_mins_idx.shape) - bimg
    plt.rcParams.update({'font.size':12})
    for i in range(len(keys)):
        lines = i
        val = np.nanquantile(tot_fit_amps[lines], 0.95) 
        print('val quantile 0.95 : ', val)
        clrimg[:,:,0] = rimg*np.clip(tot_fit_amps[lines], 0, val)/val
        clrimg[:,:,1] = gimg*np.clip(tot_fit_amps[lines], 0, val)/val
        clrimg[:,:,2] = bimg*np.clip(tot_fit_amps[lines], 0, val)/val
        plt.subplot(4,4,i+2)
        plt.imshow(clrimg[cropx:cropy,:,:]**gfac,extent=[0,160*2.5,cropx,cropy])   #4:1 extent=[0,160*2.5,125,725]
        plt.title('RGB DEM Plot - '+str(keys[lines]))
    plt.subplot(4,4,1)
    locsx, labelsx = plt.xticks()
    locsy, labelsy = plt.yticks()
    labelsx, locsx = [0, 0.5, 1], np.linspace(0,31,3)
    labelsy, locsy = [0, 25, 50, 75, 100], np.linspace(0,255,5)
    plt.xticks(locsx, labelsx), plt.xlabel('Line intensity')
    plt.yticks(locsy, labelsy), plt.ylabel('Coronal abundance (%)')
    plt.imshow(cb_img), plt.gca().invert_yaxis()
    plt.show()
    
    
def fit_lines_SPICE(filename, data_path = "SPICE_files"):

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
    tot_fit_amps = []
    tot_errors = []
    tot_summean = []

    for key in tqdm_notebook(keys) :
        raster = exposure[key]
        print(raster)
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
        fit_amps = np.zeros([nx,ny])
        fit_cens = np.zeros([nx,ny])
        fit_sigs = np.zeros([nx,ny])
        fit_cont = np.zeros([nx,ny])
        fit_err = np.zeros([nx,ny])
        intensity_summean = np.zeros([nx,ny])
        
        for i in tqdm_notebook(range(0,nx)):
            for j in range(0,ny):
                data = raster[0, :, i, j].data*u.adu
                errs = errors[0,:,i,j]
                mask = dat_mask[0,:,i,j]
                data[mask] = 0.0*u.adu
                errs[mask] = 1.0e20
                if(np.sum(np.logical_not(mask)) > 5):
                    dat = data[np.logical_not(mask)].value
                    wvl = x[np.logical_not(mask)]
                    cont = np.min(dat)
                    #amplitude, center and sigma guesses 
                    amp = np.trapz(dat-cont,x=wvl)*u.adu
                    cen = np.trapz(wvl*(dat-cont),x=wvl)*u.adu/amp
                    sig = (np.trapz(wvl**2*(dat-cont),x=wvl)*u.adu/amp-cen**2)**0.5

                    spec = Spectrum1D(flux = data, spectral_axis = x, mask=mask, uncertainty=StdDevUncertainty(errs))                    
                    g_init = models.Gaussian1D(amplitude=np.max(dat)*u.adu,
                                       mean=cen,        
                                       stddev=sig)
                    c_init = models.Const1D(amplitude = noisefloor)
                    g_fit = fit_lines(spec, g_init+c_init)
                    y_fit = g_fit(x)
                    ## Add fits to output arrays:
                    fit_cens[i][j] = (g_fit.mean_0.value)
                    fit_amps[i][j] = g_fit.amplitude_0.value
                    fit_sigs[i,j] = g_fit.stddev_0.value
                    fit_cont[i,j] = g_fit.amplitude_1.value
                    fit_err[i,j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    #to compare with an eventual mean of the rough data
                    intensity_summean[i][j] = np.nanmean(y_fit.value)
        tot_fit_amps.append(fit_amps)
        tot_errors.append(fit_err)
        tot_summean.append(intensity_summean)
    
    ##plot intensity and error maps
    if nbr_keys == 3 :   
        ################   Intensities
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1, 3),
                        axes_pad=0.05,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)
        
        for i in range(len(keys)):
            grid[i].set_axis_off()
            im = grid[i].imshow(tot_fit_amps[i]/np.quantile(tot_fit_amps[i], 0.92),extent=[0,160*2.5,125,725],cmap='gist_heat',
                                vmin=0, vmax=1)
            grid[i].set(title = str(keys[i]))
        
        cbar = grid[i].cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_ylabel('Normalized intensity')
        plt.suptitle('Intensities')
        plt.show()
        
        ####################   Errors
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1, 3),
                        axes_pad=0.05,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)
        
        for i in range(len(keys)):
            grid[i].set_axis_off()
            im = grid[i].imshow(tot_errors[i]/np.quantile(tot_errors[i],0.92),extent=[0,160*2.5,125,725],cmap='viridis',
                                vmin=0, vmax=1)
            grid[i].set(title = str(keys[i]))
        
        cbar = grid[i].cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_ylabel('Normalized error')
        plt.suptitle('Estimated errors of the fit')
        plt.show()
        
        ########## SNR
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1, 3),
                        axes_pad=0.05,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)
        
        for i in range(len(keys)):
            grid[i].set_axis_off()
            im = grid[i].imshow(tot_fit_amps[i]/tot_errors[i],extent=[0,160*2.5,125,725],cmap='plasma',
                                vmin=0, vmax=np.quantile(tot_fit_amps[i]/tot_errors[i], 0.92))
            grid[i].set(title = str(keys[i]))
        
        cbar = grid[i].cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_ylabel('SNR value')
        plt.suptitle('Signal to noise ratio')
        plt.show()
        
    else :
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1, 4),
                        axes_pad=0.05,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)
        
        for i in range(len(keys)):
            grid[i].set_axis_off()
            im = grid[i].imshow(tot_fit_amps[i]/np.quantile(tot_fit_amps[i], 0.92),extent=[0,160*2.5,125,725],cmap='gist_heat',
                                vmin=0, vmax=1)
            grid[i].set(title = str(keys[i]))
        
        cbar = grid[i].cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_ylabel('Normalized intensity')
        plt.suptitle('Intensities')
        plt.show()
        
        ####################   Errors
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1, 4),
                        axes_pad=0.05,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)
        
        for i in range(len(keys)):
            grid[i].set_axis_off()
            im = grid[i].imshow(tot_errors[i]/np.quantile(tot_errors[i],0.92),extent=[0,160*2.5,125,725],cmap='viridis',
                                vmin=0, vmax=1)
            grid[i].set(title = str(keys[i]))
        
        cbar = grid[i].cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_ylabel('Normalized error')
        plt.suptitle('Estimated errors of the fit')
        plt.show()
        
        ########## SNR
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1, 4),
                        axes_pad=0.05,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)
        
        for i in range(len(keys)):
            grid[i].set_axis_off()
            im = grid[i].imshow(tot_fit_amps[i]/tot_errors[i],extent=[0,160*2.5,125,725],cmap='plasma',
                                vmin=0, vmax=np.quantile(tot_fit_amps[i]/tot_errors[i], 0.92))
            grid[i].set(title = str(keys[i]))
        
        cbar = grid[i].cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_ylabel('SNR value')
        plt.suptitle('Signal to noise ratio')
        plt.show()
     

    ##save the amplitudes and the errors
    file_amps = 'saved_amps_' + filename + '.json'
    file_errs = 'saved_errs_' + filename + '.json'
    with open(file_amps, 'wb') as famps:
        pickle.dump(tot_fit_amps, famps)
    with open(file_errs, 'wb') as ferr:
        pickle.dump(tot_errors, ferr)
        
    return keys, tot_fit_amps, tot_errors


def fit_contribution_DEM_abundance(filename, data_path="SPICE_files"):
    [keys, tot_fit_amps, tot_errors] = fit_lines_SPICE(filename)
    [ions, wvl] = extract_ions_wvl(keys)
    file = os.path.join(data_path, filename)
    ##Contribution Functions 
    [trespsCorona, logtsCorona, exptimes] = contribution_func_spice('sun_coronal_2012_schmelz', ions, wvl)
    [trespsPhoto, logtsPhoto, exptimes] = contribution_func_spice('sun_photospheric_2015_scott', ions, wvl)
    #mix the coronal and photosphere abundances from 0% to 100%
    mixed_tresps = []
    for i in range (0,101,5):
        mix = trespsCorona*(i/100)+trespsPhoto*(1-i/100)
        mixed_tresps.append(mix)
        
    ##DEMs
    list_chi2 =[]
    datasequences = []
    em_collections =[]
    print(logtsCorona[0].shape)
    for i in mixed_tresps :
        datasequences.append(emtk.em_data_spice(file, keys, tot_fit_amps, tot_errors, logtsCorona, i))
    for k in datasequences :    
        em_collections.append(emtk.em_collection(k))

    for j in tqdm_notebook(em_collections) :
        coeffs,logts,bases,wcs,algorithm, chi2 = simple_reg_dem_wrapper(j.data())
        list_chi2.append(chi2)
        demsequence = emtk.dem_model(coeffs,logts,bases,wcs,algorithm,simple_reg_dem_wrapper)
        j.add_model(demsequence)

    [chi_mins, chi_mins_idx] = chi2_mins(list_chi2)
    
    ##Plot the coronal abundances / line intensity
    data_path = "SPICE_files"
    file = os.path.join(data_path, filename)
    plot_abundances_intensity(keys, tot_fit_amps, chi_mins, chi_mins_idx, cropx=125, cropy=725)

    return tot_fit_amps, tot_errors, chi_mins, chi_mins_idx
