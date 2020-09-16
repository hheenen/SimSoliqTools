"""
This module contains a function to obtained predefined standard surface sites
for sampling purposes for ASE created facets: 111, 100, 211, 110

"""

import numpy as np
from copy import deepcopy
from simsoliq.ase_atoms_utils import get_inds_atoms

# TODO:
# (x) function takes: slab, facet, slab-size; returns array with relevant sites (and coordination)
# (x) dictionary for sites - predetermined
# (x) automatic `multiplication of sites`
# (2) also add-in the Metal-center tags; (identical tags for closest sites)
# (x) visualization function for metal centers


def get_slab_sites(mdtraj, facet, slabsize, ind_subs=[]):
    """
      function to retrieve ASE slab based sites and adjust (to substrate) 
 
      Parameters
      ----------
      mdtraj : mdtraj-object
        mdtraj-object for which to prepare sites
      facet : str
        name of facet for which to prepare sites, i.e. '100', '111', '110', '211' 
      slabsize : tuple
        dimensions of slab
      ind_subs : list/np-array (optional)
        array to give substrate indeces
 
      Returns
      -------
      sites : dict
          dict including entries for site positions `coord` and `scaled_coord` as well
          as tags naming the coordination of each site in `tags`

    """
    ##TODO: extension to non-orthogonal cells shouldn't be such a problem
    ##TODO: split function so that ASE is only one option and one give atoms-objects or others to feed sites
    
    # get slab -- isolate substrate, analyse lattice constant scaling
    traj0 = mdtraj.get_single_snapshot(0)
    if len(ind_subs) == 0:  # TODO: also add an option to overwrite in mdtraj
        ind_subs = mdtraj._get_substrate_indices()
    slab = get_inds_atoms(traj0, ind_subs)

    # get cell
    cell = slab.get_cell()
    # warning if non-orthogonal cell
    if cell.trace() != cell.sum():
        raise Exception('only orthogonal cells included')
    xone = cell[0,0] / slabsize[0] # use x-single for scaling of basic site
    yone = cell[1,1] / slabsize[1] # use y-single for scaling of basic site

    # sites from dictionary, rescale xy of sites (only for orthogonal here)
    from simsoliq.analyze.ASEslab_site_dict import site_dict
    fd = site_dict[facet]; fac = xone/fd['x'] 
    sites = deepcopy(fd['sites']); sites[:,:2] *=fac # rescaling
    # make dictionary
    sites = {tag:[sites[i,:].tolist() for i in range(sites.shape[0]) \
        if fd['tags'][i] == tag] for tag in set(fd['tags'])}
    
    # extend sites and filter
    ny = np.around(fd['y'] / yone,0) #multiple cell size (i.e. in 111)
    nx = np.around(fd['x'] / xone,0) #multiple cell size (i.e. in 111)
    extsites = _extendsites(sites, xone*nx, yone*ny)
    # filter sites
    extsites = {p:_filter_sites_cell(extsites[p],traj0.get_cell()) \
                                                for p in extsites}
    tags, sites = _stack_site_dict(extsites)

    # adjust z coordinates for sites
    zmax = sites[:,2].max()
    vz = slab.positions[:,2].max() - zmax + 1.0
    sites[:,2] += vz
    
    scaled_sites = np.linalg.solve(slab.get_cell(complete=True).T,
                                     sites.T).T
    
    return({'coord':sites,'scaled_coord':scaled_sites,'tags':np.array(tags)})


def _stack_site_dict(psite_dict):
    """ helper function to transform site-dict to arrays"""
    # prep sites
    stags = []; sites = np.array([])
    for site in psite_dict:
        stags += [site]*psite_dict[site][:,0].size
        if sites.size == 0:
            sites = psite_dict[site]
        else:
            sites = np.vstack((sites,psite_dict[site]))
    return(stags, sites)


def _extendsites(sitedict, x, y):
    """ helper function to extend site dict far beyond unit cell"""
    cell = np.array([[x,0],[0,y]])
    shifts = [[a,b] for a in [-3,-2,-1,0,1,2,3,4,5,6] for b in [-3,-2,-1,0,1,2,3,4,5,6]]
    psitedict = {sk:[] for sk in sitedict}
    for shift in shifts:
        for sk in sitedict:
            for i in range(len(sitedict[sk])):
                pos = deepcopy(sitedict[sk][i])
                pos[:2] += np.dot(shift,cell)
                psitedict[sk].append(pos)
    psitedict = {sk:np.array(psitedict[sk]) for sk in psitedict}
    return(psitedict)


def _filter_sites_cell(sites,cell):
    """ filter site-array within bounds of `cell` """
    eps = 1e-2
    lxy = np.array([cell[0,0],cell[1,1]])
    nind = [i for i in range(sites[:,0].size) if not \
        np.any((sites[i,:2] < 0 - eps).tolist()+(sites[i,:2] - lxy >= 0 - eps).tolist())]
    return(sites[nind,:])


def visualize_sites(mdtraj, facet, slabsize, view=True):
    """
      function to visualize sites for ASE-slabs
      see parameters in `get_slab_sites`

    """
    site_data = get_slab_sites(mdtraj, facet, slabsize)
    sites = site_data['coord']
    
    traj0 = mdtraj.get_single_snapshot(0)
    ind_subs = mdtraj._get_substrate_indices()
    slab = get_inds_atoms(traj0, ind_subs)
    
    from ase.visualize import view
    from ase.atoms import Atoms
    sa = Atoms(numbers=np.ones(sites[:,0].size)*2.,positions=sites,cell=slab.get_cell())
    if view:
        view(slab+sa)
    else:
        return(slab+sa)


def get_top_slab_atoms(mdtraj, site_data={}):
    """
      function to retrieve the top atoms of a slab -- easiest way to count 
      estimate adsorption and complementary to site-model
 
      Parameters
      ----------
      mdtraj : mdtraj-object
        mdtraj-object for which to prepare sites
      site_data : dict (optional)
        dict of site-information (see `get_slab_sites`) to sort atomic 
        centers to sites
 
      Returns
      -------
      sites : dict
          dict including entries for site positions `coord` and `scaled_coord` as well
          as indeces pointing to  tags of each original site distribution

    """

    from simsoliq.geometry_utils import get_CN, _correct_vec
    # make slab to identify top row atoms 
    traj0 = mdtraj.get_single_snapshot(0)
    ind_subs = mdtraj._get_substrate_indices()
    slab = get_inds_atoms(traj0, ind_subs)

    # take CNs to identify top row
    inds, cns = get_CN(slab, rcut=3.5)
    zs = slab.get_positions()[:,2]
    imax = inds[np.where(cns == cns.max())[0]]
    isrf = inds[np.where(cns != cns.max())[0]]
    itop = [i for i in isrf if np.any(zs[imax] < zs[i])]
    itop = np.array(ind_subs)[itop]
    
    # position of metal centers 
    ptop = traj0.get_positions()[itop,:]
    sptop = traj0.get_scaled_positions()[itop,:]
    
    # if site_data given sort tags
    if len(site_data) == 0:
        raise Exception('need to implement dummy tags')
    ssites = site_data['coord']
    tagtop = []
    for j in range(ptop[:,0].size):
        dmests = np.linalg.norm(_correct_vec(ssites - ptop[j,:]),axis=1).argsort()
        for jj in dmests:
            if site_data['tags'][jj].split('-')[0] == 'top':
                tagtop.append(jj)
                break
    
    return({'coord':ptop,'scaled_coord':sptop,'itop':itop,'itags':np.array(tagtop)})

