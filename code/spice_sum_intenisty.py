"""
Created on Mon Apr  4 15:49:06 2022

@author: Tania Varesano
"""

from sunraster.instr.spice import read_spice_l2_fits
import numpy as np
from astropy import units as u

from tqdm.notebook import tqdm_notebook
import warnings 
warnings.filterwarnings("ignore") 

from scipy.ndimage import median_filter

def sum_intensity_SPICE(file):

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
    keys = ['O III 703 / Mg IX 706 - SH', 'S IV 750/ Mg IX (spectral bin 2)', 'N IV 765 - Peak', 'Ne VIII 770 / Mg VIII 772 - SH',
        'S V 786 / O IV 787 - LW', 'N III 991 - SH', 'O VI 1032 - Peak']
    shotnoise_fac = 0.025*np.sqrt(10)
    noisefloor = 0.07
    tot_sum_amps = []
    tot_sum_errs = [] 
    
    for key in keys :
        print(key)
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
        
        dat_arr = cube
        dat_filt = median_filter(dat_arr,size=3)
        filt_thold = 1.0
        dat_median = np.nanmedian(np.abs(dat_filt))
        dat_mask = (np.isnan(dat_arr) + np.isinf(dat_arr) +
                    (np.abs(dat_arr-dat_filt) > filt_thold*(dat_median+np.abs(dat_filt)))+ (dat_arr < - 0.0)) > 0
    
        errors = ((noisefloor**2+np.abs(dat_filt)*shotnoise_fac**2)**0.5).astype('float32')
        [nx,ny] = cube.shape[0:2]
        
        if key == 'O III 703 / Mg IX 706 - SH' : 
            sum_ampsOIII = np.zeros([nx,ny])
            sum_errOIII = np.zeros([nx,ny])
            sum_ampsMgIX = np.zeros([nx,ny]) 
            sum_errMgIX = np.zeros([nx,ny])
            for i in tqdm_notebook(range(0,nx)):
                for j in range(0,ny):
                    data = cube[i, j, :]*u.adu
                    errs = errors[i, j, :] 
                    mask = dat_mask[i, j, :]
                    data[mask] = 0.0*u.adu
                    if(np.sum(np.logical_not(mask)) > 5):        
                        sum_ampsOIII[i][j] = np.nanmean(cube[i,j,19:38]) # O III 703
                        sum_ampsMgIX[i][j] = np.nanmean(cube[i,j,38:]) # Mg 706
                        sum_errOIII[i][j] = np.abs((sum_ampsOIII[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                        sum_errMgIX[i][j] = np.abs((sum_ampsMgIX[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
            tot_sum_amps.append(sum_ampsOIII)
            tot_sum_amps.append(sum_ampsMgIX)
            tot_sum_errs.append(sum_errOIII)
            tot_sum_errs.append(sum_errMgIX)
            
        elif key == 'S IV 750/ Mg IX (spectral bin 2)':
            sum_ampsSIV750 = np.zeros([nx,ny])
            sum_errSIV750 = np.zeros([nx,ny])
            for i in tqdm_notebook(range(0,nx)):
                for j in range(0,ny):
                    data = cube[i, j, :]*u.adu
                    errs = errors[i, j, :]
                    mask = dat_mask[i, j, :]
                    data[mask] = 0.0*u.adu
                    if(np.sum(np.logical_not(mask)) > 5):        
                        sum_ampsSIV750[i][j] = np.nanmean(cube[i,j,:])
                        sum_errSIV750[i][j] = np.abs((sum_ampsSIV750[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
            tot_sum_amps.append(sum_ampsSIV750)
            tot_sum_errs.append(sum_errSIV750)
    
        elif key == 'N IV 765 - Peak':
            sum_ampsNIV = np.zeros([nx,ny])
            sum_errNIV = np.zeros([nx,ny])
            for i in tqdm_notebook(range(0,nx)):
                for j in range(0,ny):
                    data = cube[i, j, :]*u.adu
                    errs = errors[i, j, :]
                    mask = dat_mask[i, j, :]
                    data[mask] = 0.0*u.adu
                    if(np.sum(np.logical_not(mask)) > 5):        
                        sum_ampsNIV[i][j] = np.nanmean(cube[i,j,:])
                        sum_errNIV[i][j] = np.abs((sum_ampsNIV[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
            
            tot_sum_amps.append(sum_ampsNIV)
            tot_sum_errs.append(sum_errNIV)
    
        elif key == 'Ne VIII 770 / Mg VIII 772 - SH':
            sum_ampsNe = np.zeros([nx,ny])
            sum_errNe = np.zeros([nx,ny])
            for i in tqdm_notebook(range(0,nx)):
                for j in range(0,ny):
                    data = cube[i, j, :]*u.adu
                    errs = errors[i, j, :]
                    mask = dat_mask[i, j, :]
                    data[mask] = 0.0*u.adu
                    if(np.sum(np.logical_not(mask)) > 5):        
                        sum_ampsNe[i][j] = np.nanmean(cube[i,j,:])
                        sum_errNe[i][j] = np.abs((sum_ampsNe[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
            tot_sum_amps.append(sum_ampsNe)
            tot_sum_errs.append(sum_errNe)
    
        elif key == 'S V 786 / O IV 787 - LW' :
            sum_ampsOIV  = np.zeros([nx,ny])
            sum_errOIV = np.zeros([nx,ny])
            sum_ampsSV  = np.zeros([nx,ny])
            sum_errSV = np.zeros([nx,ny])
            for i in tqdm_notebook(range(0,nx)):
                for j in range(0,ny):
                    data = cube[i, j, :]*u.adu
                    errs = errors[i, j, :]
                    mask = dat_mask[i, j, :]
                    data[mask] = 0.0*u.adu
                    if(np.sum(np.logical_not(mask)) > 5):        
                        sum_ampsSV[i][j] = np.nanmean(cube[i,j,:22]) 
                        sum_errSV[i][j] = np.abs((sum_ampsSV[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                        sum_ampsOIV[i][j] = np.nanmean(cube[i,j,22:])
                        sum_errOIV[i][j] = np.abs((sum_ampsOIV[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
            tot_sum_amps.append(sum_ampsSV)
            tot_sum_errs.append(sum_errSV)
            tot_sum_amps.append(sum_ampsOIV)
            tot_sum_errs.append(sum_errOIV)
    
        elif key == 'N III 991 - SH' :
            sum_ampsNa  = np.zeros([nx,ny])
            sum_errNa = np.zeros([nx,ny])
            sum_ampsNIII = np.zeros([nx,ny])
            sum_errNIII = np.zeros([nx,ny])
            for i in tqdm_notebook(range(0,nx)):
                for j in range(0,ny):
                    data = cube[i, j, :]*u.adu
                    errs = errors[i, j, :]
                    mask = dat_mask[i, j, :]
                    data[mask] = 0.0*u.adu
                    if(np.sum(np.logical_not(mask)) > 5): 
                        sum_ampsNa[i][j] = np.nanmean(cube[i,j,:10])
                        sum_errNa[i][j] = np.abs((sum_ampsNa[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
                        sum_ampsNIII[i][j] = np.nanmean(cube[i,j,26:])
                        sum_errNIII[i][j] = np.abs((sum_ampsNIII[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))                    
            tot_sum_amps.append(sum_ampsNa)
            tot_sum_errs.append(sum_errNa)
            tot_sum_amps.append(sum_ampsNIII)
            tot_sum_errs.append(sum_errNIII)
    
        elif key == 'O VI 1032 - Peak' :
            sum_ampsOVI = np.zeros([nx,ny])
            sum_errOVI = np.zeros([nx,ny])
            for i in tqdm_notebook(range(0,nx)):
                for j in range(0,ny):
                    data = cube[i, j, :]*u.adu
                    errs = errors[i, j, :]
                    mask = dat_mask[i, j, :]
                    data[mask] = 0.0*u.adu
                    if(np.sum(np.logical_not(mask)) > 5):
                        sum_ampsOVI[i][j] = np.nanmean(cube[i,j,:])
                        sum_errOVI[i][j] = np.abs((sum_ampsOVI[i][j]/np.nansum(data[mask==0].value))*np.sqrt(np.nansum((errs[mask==0])**2)))
            tot_sum_amps.append(sum_ampsOVI)
            tot_sum_errs.append(sum_errOVI)
    
                        
    return tot_sum_amps, tot_sum_errs                 
                    

    
    
    
    
    
    