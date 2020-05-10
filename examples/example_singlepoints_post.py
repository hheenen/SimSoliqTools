#!/usr/bin/env python

import numpy as np

from simsoliq.io import init_mdtraj
from simsoliq.plotting.standard_plots import plot_profile

def print_singlepoint_data(key, data):
    outtxt = 5*"#" + " singlepoint data: {:>5} ".format(key) + 5*"#" + "\n"
    tkeys = list(data.keys()); tkeys.sort()
    for t in tkeys:
        outtxt += 'ts {:05d} = {:}\n'.format(t,data[t])
    outtxt += 35*"#"
    print(outtxt)
    

if __name__ == "__main__":
    
    
    ##################################################################
    ### this example demonstrates simple read-out of different     ###
    ### properties from singlepoint calculations                   ###
    ### calcultations with simsoliq:                               ###
    ###     (1) potential energy                                   ###
    ###     (2) fermi energy                                       ###
    ###     (3) vacumm potential                                   ###
    ###     (4) workfunction                                       ###
    ###     (5) charge density                                     ###
    ###     (6) electrostatic potential and plot                   ###
    ##################################################################
    
    # read MD_data path
    a = init_mdtraj("../tests/data/Pt111_24H2O_x/vasprun.xml", \
                    fmat='vasp', concmode='nested', fnest='restart')
    
    # get info about mdtraj object
    print(a)
    
    # (1) read singlepoint data: potential energy
    sp_epot = a.read_singlepoint_calculations('epot')
    print_singlepoint_data('epot',sp_epot)
    
    # (2) read singlepoint data: fermi energy
    sp_ef = a.read_singlepoint_calculations('efermi')
    print_singlepoint_data('efermi',sp_ef)
    
    # (3) read singlepoint data: vacuum potential
    sp_evac = a.read_singlepoint_calculations('evac')
    print_singlepoint_data('evac',sp_evac)
    
    # (4) read singlepoint data: workfunction
    sp_wf = a.read_singlepoint_calculations('wf')
    print_singlepoint_data('wf',sp_wf)

    # (5) read singlepoint data: charge density
    sp_chg = a.read_singlepoint_calculations('dchg')
    print("electrons in charge density: ")
    print(", ".join(["step%i: %.3f"%(k, sp_chg[k].sum()) for k in sp_chg]))

    # (6) electrostatic potential and plot
    potel = a.read_singlepoint_calculations('elpot', timesteps=[2]) # read only 1 timestep
    atoms = a.get_single_snapshot(0) # for z coordinate
    
    #     make plot
    y = potel[2]
    x = np.linspace(0,atoms.get_cell()[2,2],y.size+1)[:-1]
    xmark = [x[np.where(y == y0)[0][0]] for y0 in sp_evac[2]] # add mark vac
    plot_profile('elpot_%s'%a.get_traj_composition(), x, y, \
        xmarkings=xmark, ylabel='potential (eV)', xlabel=r'surface normal ($\AA$)')


