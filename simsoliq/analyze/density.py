"""
This module contains functions to analyze the (averaged) density
of an mdtraj object

"""

import numpy as np
from copy import deepcopy


def isolate_solvent_density(mdtraj):
    """
      return isolated density of solvent contributions only
 
      Parameters
      ----------
      density_data : dict
        dictionary of density data as obtained from mdtraj-object
 
      Returns
      -------
      density_out : dict
        dictionary of density data os solvent adjusted to substrate
    
    """
    if 'k_substrate' not in density_data:
        # since mdtraj-object not available -- take lowest coordinate from density
        e_substrate = _retrieve_substratekey_from_density(density_data)
    else:
        e_substrate = density_data['k_substrate']

    density_out = adjust_density_to_substrate(density_data, e_substrate)
    # remove non-solvent keys
    density_out['hists'] = {k:density_out['hists'][k] \
        for k in density_out['hists'] if k.find('solv') != -1}
    # align binc to zero
    density_out['binc'] -= density_out['binc'][0]
    return(density_out)


def _retrieve_substratekey_from_density(density_data):
    """ 
      helper function to retrieve substrate element from density
      density is 1-D property, substrate is assumed to be located
      at the minimum of the 1D coordinate; this is only included
      to read density data not including substrate information
    
    """
    # all density keys
    ks = list(density_data['hists'].keys())
    # fill in all minimum and maximum occurrences of density
    kind = np.zeros((len(ks),2))
    for i in range(len(ks)):
        ind = np.where(density_data['hists'][ks[i]] != 0.0)[0]
        kind[i,:] = [ind.min(), ind.max()]
    # substrate indice for minimum occurrence - flawed can only be one
    sind = kind[:,0].argmin()
    
    # make sure that substrate is isolated!
    assert np.all([kind[sind,1] < kind[k,0] for k in range(len(ks)) \
        if k != sind])
    return([ks[sind]])


def adjust_density_to_substrate(dens, esub):
    """
      helper function to align density histogram excluding substrate
    """
    # find where substrate is in density profile
    indm = max([np.where(dens['hists'][ek] != 0.0)[0] for ek in esub])
    dens['binc'] = dens['binc'][indm[-1]+1:]
    for k in dens['hists']:
        dens['hists'][k] = dens['hists'][k][indm[-1]+1:]
    return(dens)


def get_peak_integral(density_data, xlim=None, Acell=None):
    """
      return integral of density from onset until bound xlim
 
      Parameters
      ----------
      density_data : dict
        dictionary of density data as obtained from mdtraj-object
      xlim : dict (optional)
        dictionary of integration bound (in AA) for density entry
        if not specified minimum after first peak is taken
      Acell : float (optional)
        area of simulation cell perpendicular to surface normal in AA
        if given, the integral is converted to (nH2O per unit cell)
 
      Returns
      -------
      dint : dict
        dictionary with entry for each density entry containing
        integration bound and integral
    
    """
    # if no integration bounds are given, take minimum after first maximum
    if xlim == None:
        inds = _get_peak_minium_positions(density_data)
        xlim = {wk:inds[wk][1] for wk in inds}

    # convert xlim positions to indices
    ilim = {wk:np.where(density_data['binc'] >= xlim[wk])[0][0] for wk in xlim}

    # compute integral: 
    #   if Acell given (within mdtraj object) then compute nh2o per cell, else
    #   give numerical integral in mol / cm2
    d_int = {}
    for wk in ilim:
        # starting point
        istart = np.where(density_data['hists'][wk] > 0.0)[0][0]
        # integral in mol / cm2
        igr = np.trapz(density_data['hists'][wk][istart:ilim[wk]], \
            density_data['binc'][istart:ilim[wk]]) * 1e-8
        if Acell != None:
            # normalize integral to nH2O per unit cell
            nfac = {'Osolv':1., 'Hsolv':2.} # hardcoded for water
            Ncell = Acell * (1e-8**2.0) # unit-cell area in cm2
            Navg = 6.02214e23
            igr *= Ncell *Navg / nfac[wk]
        d_int.update({wk:[xlim[wk], igr]})
    return(d_int)


def _get_peak_minium_positions(density_data):
    """ 
      helper function to get hist maximum and subsequent minimum witin 2 AA
    
    """
    # ind-distance for 2 AA
    indd2 = np.where(density_data['binc']-\
        density_data['binc'][0] > 2.0)[0][0]
    x = density_data['binc']
    inds = {}
    for wk in ['Osolv', 'Hsolv']:
        ipk = density_data['hists'][wk].argmax()
        imn = ipk+density_data['hists'][wk][ipk:ipk+indd2].argmin()
        inds.update({wk:[x[ipk], x[imn]]})
    return(inds)


def get_average_solvent_bulk_density(density_data):
    """
      return the average solvent bulk density
 
      Parameters
      ----------
      density_data : dict
        dictionary of density data as obtained from mdtraj-object
 
      Returns
      -------
      davg : float
        share of solvent bulk density
    
    """
    wks = {k for k in density_data['hists'] if k.find('solv') != -1}
    # take minimum after first maximum + 1AA as starting point
    inds = _get_peak_minium_positions(density_data)
    xlim = {wk:inds[wk][1] for wk in inds}
    istart = {wk:np.where(density_data['binc'] >= xlim[wk]+1.)[0][0] for wk in wks}
    # take endpoint of density - 3 AA as finishing point
    indd3 = np.where(density_data['binc']-\
        density_data['binc'][0] > 3.0)[0][0]
    iend = {wk:np.where(density_data['hists'][wk] > 0.0)[0][-1] - indd3 for wk in wks}

    # take integral and devide through xrange
    dav = []
    dh2o = {'Osolv':1/18., 'Hsolv':2/18.} # hardcoded fro water
    for wk in wks:
        x = density_data['binc'][istart[wk]:iend[wk]]
        igr = np.trapz(density_data['hists'][wk][istart[wk]:iend[wk]], x)
        igr /= (x[-1]-x[0])
        igr /= dh2o[wk]
        dav.append(igr)
    return(np.mean(dav))

