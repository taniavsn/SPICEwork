# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:13:41 2022

@author: tania
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
plt.rcParams['image.origin'] = 'lower'

from specutils.fitting import fit_lines
from specutils.spectra import Spectrum1D

from astropy.visualization import quantity_support
quantity_support()
from fiasco import Ion

from EMToolKit.instruments.spice import contribution_func_spice
from EMToolKit.algorithms.simple_reg_dem_wrapper import simple_reg_dem_wrapper
import EMToolKit.EMToolKit_SPICE as emtk

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
def chi2_mins(list_chi2, extent):
    chi2s = np.dstack(list_chi2)
    chi_mins = np.zeros([chi2s.shape[0],chi2s.shape[1]])
    chi_mins_idx = np.zeros([chi2s.shape[0],chi2s.shape[1]])
    for i in range(chi2s.shape[0]):
        for j in range(chi2s.shape[1]):
            chi_mins[i,j] = min(list(chi2s[i,j]))
            chi_mins_idx[i,j] = list(chi2s[i,j]).index(min(list(chi2s[i,j])))
    ##PLot the chi2 min values
    plt.figure(constrained_layout=True)
    plt.imshow(chi_mins,vmax= np.nanquantile(chi_mins,0.9), extent=extent)   
    clrbr = plt.colorbar()
    clrbr.set_label('Min chi2 value')
    plt.show()

    return chi_mins, chi_mins_idx

def plot_abundances_intensity(keys, tot_fit_amps, chi_mins, chi_mins_idx, extent):   
    cb_img = clrbar_ab_int()
    plt.figure(figsize=[20,6])
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
        plt.subplot(1,5,i+2)
        plt.imshow((clrimg**gfac).transpose([1,0,2]) ,extent = extent)   
        plt.title(str(keys[lines]))
    plt.subplot(1,5,1)
    locsx, labelsx = plt.xticks()
    locsy, labelsy = plt.yticks()
    labelsx, locsx = [0, 0.5, 1], np.linspace(0,31,3)
    labelsy, locsy = [0, 25, 50, 75, 100], np.linspace(0,255,5)
    plt.xticks(locsx, labelsx), plt.xlabel('Line intensity')
    plt.yticks(locsy, labelsy), plt.ylabel('Coronal abundance (%)')
    plt.imshow(cb_img)
    plt.show()



def fit_lines_SPICE(filename, data_path=None):
    if data_path == None :
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
    
    
    # Define parameters of source, detector, and transformation: listed in the SPICE paper
    pxsz_mu = 18
    platescale_x = pxsz_mu/1.1 # Micron per arcsecond
    platescale_y = pxsz_mu/1.1 # Micron per arcsecond
    platescale_l = pxsz_mu/0.09 # Micron per Angstrom
    
    #set the errors parameters
    shotnoise_fac = 0.025*np.sqrt(10)
    noisefloor = 0.07
    bin_facs = np.array([1, 2, 1])
    tot_fit_amps = []
    tot_errors = []
    tot_summean = []
    tot_fit_cen = []
    tot_data_cen = []
    
    for key in tqdm_notebook(keys) :
        raster = exposure[key]
        print(raster)
        raster_yl = 50 #raster.data[0,15,:,:].shape[0]
        raster_yh = 800 #raster.data[0,15,:,:].shape[1]
        #downsize the raster
        cube = raster[0].data.transpose([2,1,0])[:,raster_yl:raster_yh,:]
        det_plane_min = np.nanmin(cube,axis=0)
        for i in range(0,cube.shape[0]): 
            cube[i,:,:] -= det_plane_min
        print(cube.shape)
        wl0 = (raster.meta.original_header['CRVAL3']-raster.meta.original_header['CDELT3']*raster.meta.original_header['CRPIX3'])*10
        det_scale0 = bin_facs*np.array([raster.meta.original_header['CDELT1']*platescale_x,
                                    raster.meta.original_header['CDELT2']*pxsz_mu,10*raster.meta.original_header['CDELT3']*platescale_l])
        det_origin0 = np.array([0.0,0.0,wl0*platescale_l])
        det_dims0 = np.array(cube.shape)
        waves = (det_origin0[2]+np.arange(det_dims0[2])*det_scale0[2])/platescale_l
        wcen0 = waves[np.nanargmax(np.nansum(np.nansum(cube,axis=0),axis=0))]
        
        
        raster = bindown(cube,np.round(np.array(cube.shape)/bin_facs).astype(np.int32))
        
        #mask and filter
        dat_arr = raster
        dat_filt = median_filter(dat_arr,size=3)
        filt_thold = 1.0
        sig0 = 0.3*u.Angstrom
        cen0 = wcen0*0.1*u.Angstrom
        dat_median = np.nanmedian(np.abs(dat_filt))
        dat_mask = (np.isnan(dat_arr) + np.isinf(dat_arr) +
                    (np.abs(dat_arr-dat_filt) > filt_thold*(dat_median+np.abs(dat_filt)))+ (dat_arr < - 0.0)) > 0
    
        [nx,ny] = raster[:,:,0].shape
        errors = ((noisefloor**2+np.abs(dat_filt)*shotnoise_fac**2)**0.5).astype('float32')
    
        #Create an array of corresponding wavelengths for the gaussian fit
        #x = raster.spectral_axis.to(u.nm)
    
        # Arrays to store the fit parameters:
        fit_amps = np.zeros([nx,ny])
        fit_cens = np.zeros([nx,ny])
        fit_sigs = np.zeros([nx,ny])
        fit_cont = np.zeros([nx,ny])
        fit_err = np.zeros([nx,ny])
        intensity_summean = np.zeros([nx,ny])
        data_center = np.zeros([nx,ny])
    
        for i in tqdm_notebook(range(0,nx)):
            for j in range(0,ny):
                data = raster[i, j, :]*u.adu
                errs = errors[i, j, :]
                mask = dat_mask[i, j, :]
                #data[mask] = 0.0
                #errs[mask] = 1.0e20
                if(np.sum(np.logical_not(mask)) >= 1):
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
    
                    g_fit = fit_lines(spec, g_init+c_init,window=max(wvl)-min(wvl))
                    y_fit = g_fit(waves*u.nm)
                    ## Add fits to output arrays:
                    
                    fit_amps[i][j] = g_fit.amplitude_0.value
                    fit_sigs[i,j] = g_fit.stddev_0.value
                    fit_cont[i,j] = g_fit.amplitude_1.value
                    fit_err[i,j] = np.abs((g_fit.amplitude_0.value/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                    #to compare with an eventual mean of the rough data
                    intensity_summean[i][j] = np.nanmean(y_fit.value)
                    # to compute the doppler shift
                    data_center[i][j] = cen.value
                    fit_cens[i][j] = (g_fit.mean_0.value)
        tot_fit_amps.append(fit_amps)
        tot_errors.append(fit_err)
        tot_summean.append(intensity_summean)
        tot_fit_cen.append(fit_cens)
        tot_data_cen.append(data_center)
    
    ##plot intensity and error maps
    extent = [0, cube.shape[0]*raster.wcs.wcs.cdelt[0], 0, raster.wcs.wcs.cdelt[1]*2*cube.shape[1]]
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 4),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1)
    
    for i in range(len(keys)):
        #grid[i].set_axis_off()
        im = grid[i].imshow(tot_fit_amps[i]/np.quantile(tot_fit_amps[i], 0.92),
                            extent = extent,
                            cmap='gist_heat',
                            vmin=0, vmax=1)
        grid[i].set(title = str(keys[i]))
    
    cbar = grid[i].cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.set_ylabel('Normalized intensity')
    plt.suptitle('Intensities')
    plt.show()
    
    ####################   Errors
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 4),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1)
    
    for i in range(len(keys)):
        #grid[i].set_axis_off()
        im = grid[i].imshow(tot_errors[i]/np.quantile(tot_errors[i],0.92),
                            extent = extent,
                            cmap='viridis',
                            vmin=0, vmax=1)
        grid[i].set(title = str(keys[i]))
    
    cbar = grid[i].cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.set_ylabel('Normalized error')
    plt.suptitle('Estimated errors of the fit')
    plt.show()
    
    ########## SNR
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 4),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1)
    
    for i in range(len(keys)):
        #grid[i].set_axis_off()
        im = grid[i].imshow(tot_fit_amps[i]/tot_errors[i],
                            extent = extent,
                            cmap='plasma',
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
        
    return data_path, keys, tot_fit_amps, tot_errors, tot_fit_cen, tot_data_cen, extent


def fit_contribution_DEM_abundance(filename, data_path = '.'):
    [data_path, keys, tot_fit_amps, tot_errors, tot_fit_cen, tot_data_cen, extent] = fit_lines_SPICE(filename)
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

    [chi_mins, chi_mins_idx] = chi2_mins(list_chi2, extent)
    
    ##Plot the coronal abundances / line intensity
    plot_abundances_intensity(keys, tot_fit_amps, chi_mins, chi_mins_idx, extent)

    return tot_fit_amps, tot_errors, chi_mins, chi_mins_idx, tot_fit_cen, tot_data_cen


def shift_correction2nd(array_doppler, cropped = False, crop82 = False, plot_detail=False):
    from scipy.optimize import curve_fit
    import numpy as np
    from copy import deepcopy
    
    if crop82 == False :
        array_doppler = array_doppler[30:,50,356]
    if cropped == False : 
        array_doppler = array_doppler[:,50,356]
    
    #1 degree model for longitude correction
    def func(x, a, b) :
        return a*x + b
    xlon = (np.arange(0, array_doppler.shape[0])*1.09)/3600 #get the x in degrees
    
    lista_lon = []
    listb_lon = []
    listerr_lon = []
    # Correct the longitude effect
    for i in range(array_doppler.shape[1]):
        popt, pcov = curve_fit(func, xlon, array_doppler[:,i])
        a,b = popt
        lista_lon.append(a)
        listb_lon.append(b)
        #error of the fit
        perr = np.sqrt(np.diag(pcov))
        listerr_lon.append(perr)
    
    # Latitude correction (differential rotation sun)
    def func_lat(phi, A, B, C):
        return A + B*np.sin(phi)**2+C*np.sin(phi)**4
    listA_lat = []
    listB_lat = []
    listC_lat = []
    listerr_lat = []
    #Correct the latitude effect
    xlat = (np.arange(0, array_doppler.shape[1])*4)/3600
    for i in range(array_doppler.shape[0]):
        poptlat, pcovlat = curve_fit(func_lat, xlat, array_doppler[i,:])
        A, B, C = poptlat
        listA_lat.append(A)
        listB_lat.append(B)
        listC_lat.append(C)
        #error of the fit
        perrlat = np.sqrt(np.diag(pcovlat))
        listerr_lat.append(perrlat)
    
    # Build the new array of corrected shifts
    corr_doppler = deepcopy(array_doppler)
    for j in range(array_doppler.shape[0]):
        corr_doppler[j,:] -= func_lat(xlat, np.mean(listA_lat),
                                  np.mean(np.mean(listB_lat)), np.mean(listC_lat))
    for j in range(array_doppler.shape[1]):
        corr_doppler[:,j] -= func(xlon, np.mean(lista_lon), np.mean(listb_lon))
        
    # Plot the corrected shift
    plt.figure(figsize=[6,6])
    im = plt.imshow(corr_doppler.T-np.median(corr_doppler),
                    extent = [0, corr_doppler.shape[0]*4, 0, 1.1*2*corr_doppler.shape[1]],
                    cmap=deepcopy(plt.cm.coolwarm))
    cbar = plt.colorbar(im, location='bottom')
    cbar.ax.set_xlabel('Shift')
    im.cmap.set_over('white'), im.cmap.set_under('white')
    im.set_clim(-np.nanquantile(corr_doppler,0.9), np.nanquantile(corr_doppler,0.9))
    plt.title('Corrected Doppler shift')
    
    # Optionnal : plot the estimation curves
    if plot_detail : 
        plt.figure(figsize=[10,6])
        for i in range(array_doppler.shape[1]):
            plt.scatter(xlon, array_doppler[:,i], color='gray', alpha=.01)
        plt.plot(xlon, func(xlon, np.mean(lista_lon), np.mean(listb_lon)), label='Longitude correction'), plt.ylim(-0.2,0.1)
        plt.legend(); plt.xlabel('Longitude axis (degrees)'); plt.ylabel('Shift (A)'); plt.show()
        for j in range(array_doppler.shape[0]):
            plt.scatter(xlat, array_doppler[j,:], color='green', alpha=.01)
        plt.plot(xlat, func_lat(xlat, np.mean(listA_lat), np.mean(listB_lat), np.mean(listC_lat)),label='Latitude correction'), plt.ylim(-0.2,0.1)
        plt.legend(); plt.xlabel('Latitude axis (degrees)'); plt.ylabel('Shift (A)'); plt.show()
    
    return corr_doppler, listerr_lat, listerr_lon