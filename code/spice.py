#Get the 'tresps', 'exptimes' : contribution functions
#logts are chosen to match the emission range (here logT [5,6])

#Different set of lines 
#lines = ['H I', 'H I','C III', 'Ne VIII', 'N IV']
#wvl = [1025, 1025, 977, 770, 765]

#lines = ['O III', 'O III',' Ne VIII', 'S V', 'O VI']
#wvl = [703, 703, 770, 786, 1032]

def contribution_func_spice(abund_filename, lines, wvl): 
    
    from astropy.visualization import quantity_support
    quantity_support()
    from fiasco import Ion
    from astropy import units as u
    import numpy as np

    #Specify the plasma properties
    logt_arr = np.linspace(5,6.5,41, dtype=np.double)
    Te = (10**(logt_arr - 6) )* u.MK
    ne = 1e8 * u.cm**-3

    contrib_func = []

    for x in range(len(lines)) :
        #Creating Ion object for each line
        ion = Ion(lines[x], Te, abundance_filename=abund_filename)
        contribution_func = ion.contribution_function(ne)
        wlen = wvl[x] * u.Angstrom

        transitions = ion.transitions.wavelength[~ion.transitions.is_twophoton]
        #index : finding the closest transition regarding the wvl specified
        idx = np.argmin(np.abs(transitions - wlen))
        contrib_func.append(contribution_func.value[:,0,idx])
        
    tresps = np.array(contrib_func)
    print('Shape tresps : ', tresps.shape)
    logts = []
    for i in range(tresps.shape[0]):
        logts.append(logt_arr)
    print('Length logts : ', len(logts))
    #Time exposure for each spectral line
    exptimes = np.array([20,20,20,20,20])
    
    return tresps, logts, exptimes