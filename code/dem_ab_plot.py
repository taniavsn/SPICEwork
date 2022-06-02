# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:34:10 2022

@author: tania
"""

import os.path
import numpy as np
from sunraster.instr.spice import read_spice_l2_fits
import matplotlib.pyplot as plt

from astropy import units as u
from tqdm.notebook import tqdm_notebook
import pickle
import warnings 
from copy import deepcopy
warnings.filterwarnings("ignore")

import EMToolKit.instruments.spice_functions_abundance as sfab
from EMToolKit.instruments.spice import contribution_func_spice
from mpl_toolkits.axes_grid1 import AxesGrid
import EMToolKit.EMToolKit_SPICE as emtk
from EMToolKit.algorithms.simple_reg_dem_wrapper import simple_reg_dem_wrapper
from ndcube import NDCube, NDCubeSequence, NDCollection
plt.rcParams['image.origin'] = 'lower'
plt.rcParams.update({'font.size': 16}) # Make the fonts in figures big enough for papers
plt.rcParams.update({'figure.figsize':[15,7]});
from multiprocessing import Pool
import matplotlib as mlt
mlt.rc('xtick', labelsize=18)
mlt.rc('ytick', labelsize=18)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

gfac=1/2.2
data_path = "mosaic"
#In DEM_abundace, "file" must contain the path, and the extensiom 
#(ex : mosaic\\solo_L2_spice-n-ras_20220302T004014_V03_100663682-000.fits)
# list_file_amps == 1 if not binned down, 2 if binned down.
def DEM_abundance(list_file_amps_err, defVal=0.01):
    
    file = list_file_amps_err[0]
    amps = list_file_amps_err[1]
    errors = list_file_amps_err[2]
    if list_file_amps_err[3] == 1: 
        extent = [0, amps[0].shape[0]*4, 0, 1.1*amps[0].shape[1]]
    if list_file_amps_err[3] == 2: 
        extent = [0, amps[0].shape[0]*4, 0, 1.1*2*amps[0].shape[1]]
    # keys = ['O III 703 / Mg IX 706 - SH', 'O III 703 / Mg IX 706 - SH','S IV 750/ Mg IX (spectral bin 2)', 'N IV 765 - Peak',
    #     'Ne VIII 770 / Mg VIII 772 - SH',  'S V 786 / O IV 787 - LW',
    #     'S V 786 / O IV 787 - LW', 'N III 991 - SH', 'N III 991 - SH', 'O VI 1032 - Peak']

    # ions = ['O III', 'Mg IX', 'S IV', 'N IV', 'Ne VIII', 'S V', 'O IV', 'Na VI', 'N III', 'O VI']
    # wvl = [703, 706, 750, 765, 770, 786, 787, 988.6, 991, 1032]
    
    keys = ['O III 703 / Mg IX 706 - SH', 'N IV 765 - Peak', 'Ne VIII 770 / Mg VIII 772 - SH', 'O VI 1032 - Peak']

    ions = ['Mg IX', 'N IV', 'Ne VIII', 'O VI']
    wvl = [706,  765, 770, 1032]    
    
    exposure = read_spice_l2_fits(file,memmap=False)
    
    ##Contribution Functions 
    [trespsCorona, logtsCorona, exptimes] = contribution_func_spice('sun_coronal_2012_schmelz', ions, wvl)
    
    #Change any values that could cause a linalg Errors
    tot_err_nonan = []
    tot_amp_nonan = []
    
    for k in range(len(amps)):
        B = np.nan_to_num( np.array(errors[k]), nan = np.nanmax(errors[k]),
                          posinf=np.nanmax(errors[k]), neginf=np.nanmax(errors[k]))
        B[ B == 0] = np.nanmax(errors[k])
        tot_err_nonan.append(B)
        A = np.nan_to_num(np.array(amps[k]), nan = defVal,
                          posinf=defVal, neginf=defVal)
        A[ A == 0] = np.nanmin(amps[k])
        tot_amp_nonan.append(A)
    
    #Tranforming arrays into NDCube objects
    for i in range((len(tot_amp_nonan))):
        raster = exposure[keys[i]]
        tot_amp_nonan[i] = NDCube(tot_amp_nonan[i], wcs = raster[0,0].wcs, meta = {"detector": "SPICE",
                                                                                   "wave_str" :list(exposure.keys())[i],
                                                                                   "exptime": 20})
    #set range of temperatures
    logt_arr = np.arange(4.5,6.55,0.05)
    logts = []
    for i in range(trespsCorona.shape[0]):
        logts.append(logt_arr)
    
    #Time exposure for each spectral line
    exptimes = np.array([20,20,20,20,20])
    
    #Turning list of fitted amplitude into usable object (datasequence)
    datasequence = emtk.em_data(tot_amp_nonan, tot_err_nonan, logts, trespsCorona)
    em_collection = emtk.em_collection(datasequence)
    
    #DEM computation
    coeffs,logts,bases,wcs,algorithm, chi2 = simple_reg_dem_wrapper(em_collection.data())
    demsequence = emtk.dem_model(coeffs,logts,bases,wcs,algorithm,simple_reg_dem_wrapper)
    em_collection.add_model(demsequence)

    gfac = 1.0/2.2
    
    #Plot the DEMs
    tempindx = [0,5,10,15,20,25]
    
    fig = plt.figure(figsize=(20, 17), constrained_layout=True)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2, 3),
                    axes_pad=0.9,
                    cbar_mode='each',
                    cbar_location='bottom',
                    cbar_pad=0.3)
    
    for i in range(6):
        im = grid[i].imshow((demsequence[tempindx[i],:,:].data.T)**gfac, 
                            extent = extent, #vmax=3e10,
                            cmap=plt.get_cmap('gray'))
        grid[i].set(title = 'DEM at Log_10(T) = '+ "%.2f" % demsequence[tempindx[i]].meta['logt0'])
        cbar = grid[i].cax.colorbar(im)
        cbar.ax.set_xlabel('DEM value')
    plt.show()
    
    ## Abundance plots : mix the abundance values and find the mix that 
    #gives the best chi2 for each pixel
    
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
    for i in mixed_tresps :
        datasequences.append(emtk.em_data_spice(file, keys, amps, errors, logtsCorona, i, defaultValue=defVal))
    for k in datasequences :    
        em_collections.append(emtk.em_collection(k))
    
    for j in tqdm_notebook(em_collections) :
        coeffs,logts,bases,wcs,algorithm, chi2 = simple_reg_dem_wrapper(j.data())
        list_chi2.append(chi2)
        demsequence = emtk.dem_model(coeffs,logts,bases,wcs,algorithm,simple_reg_dem_wrapper)
        j.add_model(demsequence)
    
    [chi_mins, chi_mins_idx] = sfab.chi2_mins(list_chi2, extent)
    
    ##Plot the coronal abundances / line intensity
    cb_img = sfab.clrbar_ab_int()
    gfac=1/2.2
    clrimg = np.zeros([chi_mins.shape[0], chi_mins.shape[1], 3])
    gimg = np.ones(chi_mins_idx.shape)
    bimg = chi_mins_idx/20
    rimg = np.ones(chi_mins_idx.shape) - bimg
    
    for i in range(len(keys)):
        plt.figure(figsize=(7,7), tight_layout=True)
        val = np.nanquantile(amps[i], 0.999) 
        #print('val quantile 0.95 : ', val)
        clrimg[:,:,0] = rimg*np.clip(amps[i], 0, val)/val
        clrimg[:,:,1] = gimg*np.clip(amps[i], 0, val)/val
        clrimg[:,:,2] = bimg*np.clip(amps[i], 0, val)/val
        plt.imshow((clrimg**gfac).transpose([1,0,2]),   
                   extent = extent)   
        plt.title(ions[i])
        plt.show()
    locsx, labelsx = plt.xticks()
    locsy, labelsy = plt.yticks()
    labelsx, locsx = [0, 0.5, 1], np.linspace(0,31,3)
    labelsy, locsy = [0, 25, 50, 75, 100], np.linspace(0,255,5)
    plt.xticks(locsx, labelsx), plt.xlabel('Line intensity')
    plt.yticks(locsy, labelsy), plt.ylabel('Coronal abundance (%)')
    plt.imshow(cb_img)
    plt.show()
    
    return chi_mins, chi_mins_idx
        

