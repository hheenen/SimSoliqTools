#!/usr/bin/env python

import sys, os
import numpy as np
from ase.io import read, write
from ase.visualize import view
from ase.build import fcc100, fcc111, fcc110, fcc211
from ase.build import sort
from ase.data import chemical_symbols
from ase.atoms import Atoms
from ase.constraints import FixAtoms

### Functions ###
#############################################################
################## NOTE: already worked in ##################
##############################################################
## from cathelpers.atom_manipulators import get_type_atoms
#def get_type_indices(atoms,atom_types):
#    ind_types = np.hstack([np.where(atoms.get_atomic_numbers() == t)[0]\
#                for t in atom_types])
#    return(ind_types)
#
#def get_type_atoms(atoms,atom_types):
#    ind_types = get_type_indices(atoms,atom_types)
#    return(get_inds_atoms(atoms,ind_types))
#
#def get_inds_atoms(atoms,ind_types):
#    return(Atoms(numbers=atoms.get_atomic_numbers()[ind_types],\
#                 positions=atoms.get_positions()[ind_types,:],\
#                 cell=atoms.get_cell()))
#
#def identify_water(atoms):
#    pos_ads = atoms.get_scaled_positions()
#    cell = atoms.get_cell()
#    ind_O = get_type_indices(atoms,[8])
#
#    # check neighbors in all types / could equally be only H-type
#    type_res = list(set((atoms.get_atomic_numbers())))
#    type_res.remove(8)
#    ind_rest = get_type_indices(atoms,type_res)
#    type_R = atoms.get_atomic_numbers()[ind_rest]
#    pos_R = pos_ads[ind_rest,:]
#    # TODO: may become problematic - typeR only H, slowly increase R and remove closest O-H distances
#    water_like = {}
#    for o in ind_O:
#        pos_O = pos_ads[o,:]
#        d_vec = np.linalg.norm(np.dot(_correct_vec(pos_R - pos_O),cell),axis=1)
#        ind_neigh = np.where(d_vec < 1.3)[0]
#        if sum(type_R[ind_neigh]) == 2: #ONLY H2O
#            water_like.update({o:ind_rest[ind_neigh]})
#    return(water_like)
    
#def _correct_vec(vec):
#    ''' correct vectors in fractional coordinates 
#        (assuming vectors minimal connection between 2 points)
#    '''
#    vec[np.where(vec >= 0.5)] -= 1.0
#    vec[np.where(vec < -0.5)] += 1.0
#    return(vec)
#############################################################
#############################################################
##############################################################

# dream script:

# (1) recognize input slab
#   (a) recognize surface shape
#   (b) recognize xy dimensions, layer depth
#   (c) recognize orientation - use ASE and turn/shift... 
#                               adjust adsorbate position
# (2) place adsorbate and water from original cell
#   (a) option: add water layer
#   (b) option: set water-vacuum distance

def adjust_cells(refs,surf,xyz,element,lat,\
    height=None,vacuum=None,add_water_layer=False,n_p_layer=None):
    # set-up new slab - super large slab to put water - adjustment later
    slab = fcc211(element, a=lat, size=tuple(xyz), vacuum=12.0, orthogonal=True)
    # orientation can be skipped - Tom's structures also generated with ASE

    an_element = chemical_symbols.index(element)
    an_adsorbs = [a for a in np.unique(refs.get_atomic_numbers()) \
                                                if a != an_element]

    adsorbs = get_type_atoms(refs,an_adsorbs)
   
    # identiy and adjust distance for 'centered' new slab and adsorbates+water
    com_surf0 = get_type_atoms(refs,[an_element]).get_center_of_mass()
    com_ads = get_type_atoms(refs,an_adsorbs).get_center_of_mass()
    d_SA = com_ads[2] - com_surf0[2]
    com_surf = get_type_atoms(slab,[an_element]).get_center_of_mass()
    d_SA1 = com_ads[2] - com_surf[2]
    # adjust
    pos0 = adsorbs.get_positions()
    pos0[:,2] += d_SA - d_SA1
    adsorbs.set_positions(pos0)
    
    # if adding a nother water layer - copy paste top layer
    if add_water_layer:
        if n_p_layer == None:
            raise Exception("no input for number of waters per layer")
        adsorbs = add_water_layer_to_bilayer(adsorbs,n_p_layer)

    # put vacuum
    adjusted_slab = slab+adsorbs
    if height != None:
        pcell = adjusted_slab.get_cell()
        adjusted_slab.set_cell([pcell[0][0],pcell[1][1],height])
        adjusted_slab.center(axis=2)
    elif vacuum != None:
        adjusted_slab.center(axis=2,vacuum=vacuum)
    else:
        raise Exception("either height or vacuum must be defined")
    adjusted_slab = sort(adjusted_slab, \
                    tags=adjusted_slab.get_atomic_numbers())
    # note - leave last layer non-fixed = upper 3 z coords
    pos = adjusted_slab.get_positions()
    indMe = np.where(adjusted_slab.get_atomic_numbers() == an_element)[0]
    zll = np.unique(pos[indMe,2])[-3:]
    indll = [i for i in indMe if pos[i,2] not in zll]
    c = FixAtoms(indices=indll)
    adjusted_slab.set_constraint(c)
    
    return(adjusted_slab)

def add_water_layer_to_bilayer(adsorbs,n_p_layer):
    # water identifier is indice of oxygen atom
    pos_ads = adsorbs.get_positions()
    ind_O = get_type_indices(adsorbs,[8])
    order_O = ind_O[pos_ads[ind_O,2].argsort()] #ordered
    
    # now identify water molecules:
    wlike = identify_water(adsorbs)
    order_O = np.array([o for o in order_O if o in wlike])
    
    # since known: missing water is accounted to firs layer (ion)
    missO = len(order_O)%n_p_layer
    if missO != 0:
        order_O = np.array([[order_O[0]]*(n_p_layer-missO) + order_O.tolist()])
    
    # identify distance between last two rows
    order_O = np.reshape(order_O,(int(order_O.size/n_p_layer),n_p_layer))
    indr0 = np.array([[a]+[b for b in wlike[a]] for a in order_O[-2,:].tolist()]).flatten()
    indr1 = np.array([[a]+[b for b in wlike[a]] for a in order_O[-1,:].tolist()]).flatten()
    d_wl = (get_inds_atoms(adsorbs,indr1).get_center_of_mass() - \
           get_inds_atoms(adsorbs,indr0).get_center_of_mass())[2]
    
    # add water-layer
    extr_ly = get_inds_atoms(adsorbs,indr1)
    epos = extr_ly.get_positions()
    epos[:,2] += d_wl
    extr_ly.set_positions(epos)

    adsorbs = adsorbs + extr_ly
    return(adsorbs)


if __name__ == "__main__":

    ref_struc = 'test_strucs/Cu211_10H2O_OCCHO.traj'
    ref_struc = sys.argv[1]
    filename = ref_struc.split('/')[-1]

    refs = read(ref_struc)
    
    # presets for this study (and 3x3 cell)
    #rpe_lat = 3.5733446 #ENCUT=500eV
    rpe_lat = 3.56878996 #ENCUT=400eV
    
    # automatic recognition in principle
    surf = '211'
    xyz = [3,3,3]
    element = 'Cu'
    n_p_layer = 5
    vacuum = 5 # for diple cells

    # options: add water layer
    add_water_layer = sys.argv[2]

    b = adjust_cells(refs,\
            surf,\
            xyz,\
            element,\
            rpe_lat,\
            vacuum=vacuum,\
            add_water_layer=add_water_layer,\
            n_p_layer=n_p_layer)

    fn = filname.split('.')
    write(fn[0]+"_adjusted."+fn[1],b)

