import copy
import numpy as np
from sunpy.map import Map
from ndcube import NDCube, NDCubeSequence, NDCollection
from astropy.coordinates import SkyCoord
from astropy.nddata import StdDevUncertainty
import astropy.wcs
from sunraster.instr.spice import read_spice_l2_fits
from astropy.nddata import UnknownUncertainty
from tqdm.notebook import tqdm_notebook as tqdm

#for em_data, input should be the amplitude fits and the errors from the spectral line fitting
#amps, errs, should be lists of length n_channels

def em_data_spice(file, keys, amps, errs, logts, tresps, channels=None, defaultValue=1):
    
    exposure = read_spice_l2_fits(file,memmap=False)
    #keys must be the full list or a part of the list of exposure.keys()
    #get rid of any values that could cause a LinAlg error
    tot_err_nonan = []
    tot_amp_nonan = []
    for k in range(len(amps)):
        
        A = np.array(amps[k])
        A = np.nan_to_num(A)
        A[ A == np.inf] = defaultValue
        tot_amp_nonan.append(A)
        
        B = np.array(errs[k])
        B[np.isnan(B)] = defaultValue
        B[ B == np.inf] = defaultValue
        B[ B == 0] = defaultValue
        tot_err_nonan.append(B)
   
    amps = tot_amp_nonan
    errs = tot_err_nonan
    
    #add wcs data and transforming into NDCubes
    for i in range(len(amps)):
        raster = exposure[keys[i]]
        amps[i] = NDCube(amps[i], wcs = raster[0,0].wcs, meta = {"detector": "SPICE",
                                                               "wave_str" : keys[i],
                                                               "exptime": 20})
    
    cubes = []
    for i in range(0,len(amps)):
        ampsi = copy.copy(amps[i])
        if('channel' not in ampsi.meta):
            if(channels is None): ampsi.meta['channel'] = ampsi.meta['detector']+ampsi.meta['wave_str']
            else: ampsi.meta['channel'] = channels[i]
        ampsi.meta['logt'] = logts[i]
        ampsi.meta['tresp'] = tresps[i]
        cubes.append(NDCube(ampsi,uncertainty=errs[i]))
    return NDCubeSequence(cubes)


def em_data(maps,errs,logts,tresps,channels=None):
    cubes = []
    for i in range(0,len(maps)):
        mapi = copy.copy(maps[i])
        if('channel' not in mapi.meta):
            if(channels is None): mapi.meta['channel'] = mapi.meta['detector']+mapi.meta['wave_str']
            else: mapi.meta['channel'] = channels[i]
        mapi.meta['logt'] = logts[i]
        mapi.meta['tresp'] = tresps[i]
        cubes.append(NDCube(mapi,uncertainty=errs[i]))
    return NDCubeSequence(cubes)
    
    
    
def dem_model(coeffs,logts,bases,wcs,algorithm,wrapper,meta={},wrapargs=None):

    nd = len(coeffs)
    dem_sequence = []
    for i in range(0,nd):
        logt0 = np.median(logts[i][np.where(bases[i]==np.max(bases[i]))])
        metai = {'basis':bases[i],'logt':logts[i],'logt0':logt0,'algorithm':algorithm,'wrapper':wrapper,'wrapargs':wrapargs}
        dem_sequence.append(NDCube(coeffs[i],wcs=wcs,meta={**meta,**metai}))
    return NDCubeSequence(dem_sequence,meta={'algorithm':algorithm,'wrapper':wrapper,'wrapargs':wrapargs})
    
class em_collection:

    def __init__(self,datasequence):
        self.collection = NDCollection([("data",datasequence),("models",[])])

    def data(self): return self.collection['data']

    def add_model(self,modelsequence):
        # Have to remake the collection from keys because update doesn't work when there are no aligned axes
        # and we can't use aligned axes because all parts of the collection must have aligned axes, which is
        # not necessarily true for the elements of the em collection!!! Obviously we'll want to change this
        # back once those issues are fixed on the NDCube side.
        pairs = [] 
        for k in self.collection.keys(): pairs.append((k,self.collection.get(k)))
        pairs.append((modelsequence.meta['algorithm'],modelsequence))
        self.collection.clear()
        self.collection = NDCollection(pairs)
        self.collection['models'].append(modelsequence.meta['algorithm'])

    def compute_dem(self,i,j,logt=None,algorithm=None):
        from scipy.interpolate import interp1d

        if(algorithm is None): algorithm = self.collection['models'][0]
        model = self.collection[algorithm]
        if(logt is None): logt = model[0].meta['logt']
        dem = np.zeros(logt.size)
        for component in model:
            complogt = component.meta['logt']
            compbasis = component.meta['basis']
            dem += component.data[i,j]*interp1d(complogt,compbasis,fill_value=0.0,bounds_error=False)(logt)
        return logt,dem	

    def synthesize_data(self, logts, tresps, algorithm=None, channels=None, ilo=0, ihi=-1, jlo=0, jhi=-1):
        from scipy.integrate import trapezoid
        from scipy.interpolate import interp1d
        if(logts[0].size == 1): [logts, tresps] = [[logts],[tresps]]
        ndata = len(logts)
        if(algorithm is None): algorithm = self.collection['models'][0]
        if(channels==None): channels = ['SYNTHDATA'+str(i) for i in range(0,ndata)]
        model = self.collection[algorithm]
        [synthmaps,syntherrs] = [[],[]]
        for i in range(0, ndata):
            synthdata = np.zeros(model[0].data[ilo:ihi,jlo:jhi].shape)
            syntherrs.append(UnknownUncertainty(np.zeros(model[0].data.shape)-1))
            synthmap = copy.deepcopy(model[0])[ilo:ihi,jlo:jhi]
            synthmap.meta['algorithm'] = algorithm
            synthmap.meta['channel'] = channels[i]
            datainterp = interp1d(logts[i], tresps[i], fill_value=0.0, bounds_error=False)
            for component in model:
                basistemp = component.meta['logt']
                basisinterp = interp1d(basistemp, component.meta['basis'], fill_value=0.0, bounds_error=False)
                logt = np.unique(np.hstack([basistemp, logts[i]]))
                coupling = trapezoid(datainterp(logt)*basisinterp(logt),x=logt)
                synthdata += coupling*component.data[ilo:ihi,jlo:jhi]
            synthmap.data[:] = synthdata[:]
            synthmaps.append(synthmap)

        return em_data(synthmaps,syntherrs,logts,tresps,channels=channels)