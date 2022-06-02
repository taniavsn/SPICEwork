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
    plt.imshow(chi_mins_idx.T,vmax= np.nanquantile(chi_mins,0.9), extent=extent)   
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


# def fit_contribution_DEM_abundance(filename, data_path = '.'):
#     [data_path, keys, tot_fit_amps, tot_errors, tot_fit_cen, tot_data_cen, extent] = fit_lines_SPICE(filename)
#     [ions, wvl] = extract_ions_wvl(keys)
#     file = os.path.join(data_path, filename)
#     ##Contribution Functions 
#     [trespsCorona, logtsCorona, exptimes] = contribution_func_spice('sun_coronal_2012_schmelz', ions, wvl)
#     [trespsPhoto, logtsPhoto, exptimes] = contribution_func_spice('sun_photospheric_2015_scott', ions, wvl)
#     #mix the coronal and photosphere abundances from 0% to 100%
#     mixed_tresps = []
#     for i in range (0,101,5):
#         mix = trespsCorona*(i/100)+trespsPhoto*(1-i/100)
#         mixed_tresps.append(mix)
        
#     ##DEMs
#     list_chi2 =[]
#     datasequences = []
#     em_collections =[]
#     print(logtsCorona[0].shape)
#     for i in mixed_tresps :
#         datasequences.append(emtk.em_data_spice(file, keys, tot_fit_amps, tot_errors, logtsCorona, i))
#     for k in datasequences :    
#         em_collections.append(emtk.em_collection(k))

#     for j in tqdm_notebook(em_collections) :
#         coeffs,logts,bases,wcs,algorithm, chi2 = simple_reg_dem_wrapper(j.data())
#         list_chi2.append(chi2)
#         demsequence = emtk.dem_model(coeffs,logts,bases,wcs,algorithm,simple_reg_dem_wrapper)
#         j.add_model(demsequence)

#     [chi_mins, chi_mins_idx] = chi2_mins(list_chi2, extent)
    
#     ##Plot the coronal abundances / line intensity
#     plot_abundances_intensity(keys, tot_fit_amps, chi_mins, chi_mins_idx, extent)

#     return tot_fit_amps, tot_errors, chi_mins, chi_mins_idx, tot_fit_cen, tot_data_cen


def shift_correction2nd(array_doppler, cropped = False, crop82 = False, plot_detail=False, rot_corr=False):
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
    
    corr_doppler = deepcopy(array_doppler)
    for j in range(array_doppler.shape[1]):
        corr_doppler[:,j] -= func(xlon, np.mean(lista_lon), np.mean(listb_lon))    
        
    ##### OPTIONNAL : correct the diffrential rotation of the Sun 
    if rot_corr == True :
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
    plt.figure(figsize=[16,6])
    plt.subplot(1,3,1)
    im = plt.imshow(corr_doppler.T-np.median(corr_doppler),
                    extent = [0, corr_doppler.shape[0]*4, 0, 1.1*2*corr_doppler.shape[1]],
                    cmap=deepcopy(plt.cm.coolwarm))
    cbar = plt.colorbar(im, location='bottom')
    cbar.ax.set_xlabel('Shift (A)')
    im.cmap.set_over('white'), im.cmap.set_under('white')
    im.set_clim(-np.nanquantile(corr_doppler,0.9), np.nanquantile(corr_doppler,0.9))
    plt.title('Corrected Doppler shift'), plt.xlabel('Longitude axis (arcsec)'), plt.ylabel('Latitude axis (arcsec)')

    plt.subplot(1,3,2)
    im = plt.imshow(array_doppler.T,
                    extent = [0, corr_doppler.shape[0]*4, 0, 1.1*2*corr_doppler.shape[1]],
                    cmap=deepcopy(plt.cm.coolwarm))
    cbar = plt.colorbar(im, location='bottom')
    cbar.ax.set_xlabel('Shift (A)')
    im.cmap.set_over('white'), im.cmap.set_under('white')
    im.set_clim(-np.nanquantile(array_doppler,0.85), np.nanquantile(array_doppler,0.85))
    plt.title('Original Doppler shift'), plt.xlabel('Longitude axis (arcsec)')
    
    plt.subplot(1,3,3)
    diff = array_doppler.T-(corr_doppler.T-np.median(corr_doppler))
    im = plt.imshow(diff,
                    extent = [0, corr_doppler.shape[0]*4, 0, 1.1*2*corr_doppler.shape[1]],
                    cmap=deepcopy(plt.cm.coolwarm))
    cbar = plt.colorbar(im, location='bottom')
    cbar.ax.set_xlabel('Angstrom')
    im.set_clim(-np.nanquantile(diff,0.9), np.nanquantile(diff,0.9))
    plt.title('Difference'), plt.xlabel('Longitude axis (arcsec)')
    plt.show()
    
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
    
    return corr_doppler