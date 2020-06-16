#!/usr/bin/env python

import numpy as np
from ase.data import chemical_symbols as symbols
from simsoliq.io import init_mdtraj
from simsoliq.plotting.standard_plots import plot_density
from simsoliq.analyze.density import isolate_solvent_density, get_peak_integral, \
    get_average_solvent_bulk_density
from simsoliq.mdtraj_average import average_densities

if __name__ == "__main__":

    ##################################################################
    ### this example demonstrates the handling of density profiles ###
    ### with simsoliq:                                             ###
    ###     (1) automated creation of density plots of solvent     ###
    ###     (2) automated analysis of density of solvent           ###
    ###     (3) averaging of density from different trajectories   ###
    ###     (4) handling and plotting of raw densities from mdtraj ###
    ##################################################################


    ##################################################################
    ###     (1) automated creation of density plots of solvent     ###
    ##################################################################

    # load trajectory, of Pt(111) with 24 H2O and *OH adsorbate
    a = init_mdtraj("../tests/data/Pt111_24H2O_OH_long/vasprun.xml")
    
    # get density data from trajectory
    densdata = a.get_density_profile(height_axis=2, savepkl=True)
    
    # truncate density to solvent only
    dens = isolate_solvent_density(densdata)
    
    # plot truncated density
    plot_density('density_plot1_longMD', dens['binc'], dens['hists'])
    
    
    ##################################################################
    ###     (2) automated analysis of density of solvent           ###
    ##################################################################
    
    # compute numerical integral in mol/cm2 of density until ...
    dint_m = get_peak_integral(dens, xlim=None) # minimum after first maximum
    dint_d = get_peak_integral(dens, xlim={'Osolv':4,'Hsolv':4}) # x == 4 AA
    # .. and plot
    plot_density('density_plot2_longMD_intmin', dens['binc'], dens['hists'], dint_m)
    plot_density('density_plot3_longMD_intdist', dens['binc'], dens['hists'], dint_d)

    # compute integral normalized to number of H2O per unit cell
    Acell = a.get_cell_area() # required cell area
    dint_n = get_peak_integral(dens, xlim=None, Acell=Acell)
    # .. and plot
    plot_density('density_plot4_longMD_intnorm', dens['binc'], dens['hists'], dint_n)

    # get average density of bulk region
    dbulk = get_average_solvent_bulk_density(dens)
    
    # print messages
    print("integrated Osolv:") 
    print("minimum after first maximum at %.2f AA: %.1e mol/cm2"%tuple(dint_m['Osolv']))
    print("integral O until distance of %.2f AA: %.1e mol/cm2"%tuple(dint_d['Osolv']))
    print("minimum after first maximum at %.2f AA: %.1e n(H2O)/unit cell"%tuple(dint_n['Osolv']))
    print("average bulk density after first peak is %.2f of exp density"%dbulk)
    
    
    ##################################################################
    ###     (3) averaging of density from different trajectories   ###
    ##################################################################
    
    # read trajectories (use restart as individual traj)
    trj1 = init_mdtraj("../tests/data/Pt111_24H2O_x/vasprun.xml", fmat='vasp')
    trj2 = init_mdtraj("../tests/data/Pt111_24H2O_x/restart/vasprun.xml", fmat='vasp')
    
    # average densities of two trajectories, `average_densities` 
    # will return an average per composition 
    # (only one composition included here)
    av_dens = average_densities([trj1, trj2], tstart=0)
    for comp in av_dens:
        c_dens = isolate_solvent_density(av_dens[comp])
        binc = c_dens['binc']
        hist = c_dens['hists']
        plot_density('density_plot5_average_%s'%comp, binc, hist)

    
    ##################################################################
    ###     (4) handling and plotting of raw densities from mdtraj ###
    ##################################################################

    # load trajectory, of Pt(111) with 24 H2O and *OH adsorbate
    a = init_mdtraj("../tests/data/Pt111_24H2O_OH_long/vasprun.xml")
    
    # get density data
    densdata = a.get_density_profile(height_axis=2, savepkl=False)
    binc = densdata['binc']
    hist_dicts = densdata['hists']
    hist_dicts = {el:hist_dicts[el] for el in hist_dicts \
        if el in ['Pt', 'Osolv', 'Hsolv']}

    ### plot density of everything in the simulation
    plot_density('density_plot6_example_raw', binc, hist_dicts)
    
    ### plot density of solvent
    # (1) truncate data to relevant range by hand
    istart = np.where(hist_dicts['Pt'] > 0.0)[0][-1]+1
    # reduce data to solvent
    hist_solv = {key:hist_dicts[key][istart:] for key in hist_dicts \
        if key.find('solv') != -1}
    
    plot_density('density_plot7_solvent_only_1', binc[istart:], hist_solv)


