#!/usr/bin/env python

import numpy as np
from simsoliq.io import init_mdtraj
from simsoliq.mdtraj_average import average_energies, sort_energies, average_densities
from simsoliq.plotting.ensemble_plots import plot_running_av_ensemble
from simsoliq.plotting.standard_plots import plot_density


if __name__ == "__main__":
    # read trajectories (use restart as individual traj)
    trj1 = init_mdtraj("../tests/data/Pt111_24H2O_x/vasprun.xml", fmat='vasp')
    trj2 = init_mdtraj("../tests/data/Pt111_24H2O_x/restart/vasprun.xml", fmat='vasp')

    ###################################################################
    # mean and standard deviation of trajectories sorted by composition
    # (only one composition included here)
    e_val = average_energies([trj1, trj2], tstart=0)
    for ek in e_val:
        print(ek, e_val[ek])
    ###################################################################

    
    
    ###################################################################
    # return data for energies and plot all running averages
    edat = sort_energies([trj1, trj2])
    # artificially elongate data for more interesting plot
    for ek in edat:
        if ek not in ['timestep','timeunit']:
            for c in edat[ek]:
                for i in range(len(edat[ek][c])):
                    edat[ek][c][i] = np.array(1000*edat[ek][c][i].tolist())
    # add artificial entry for different systems
    for ek in edat:
        if ek not in ['timestep','timeunit']:
            edat[ek].update({"Au36_24H2O":[edat[ek]['Pt36_24H2O'][i] + 150]})
            edat[ek].update({"Pt36_24H2O_CO":[edat[ek]['Pt36_24H2O'][i] -1.0]})

    # plot running averages of total energy- starting averaging after tstart
    timedat = {k:edat[k] for k in ['timestep','timeunit']}
    plot_running_av_ensemble("ensemble_running_averages_etot", edat['etot'], \
        surftag = {}, tstart=0.5, tunit='ps', **timedat)
    ###################################################################

    
    
    ###################################################################
    # average densities of two trajectories, `average_densities` 
    # will return an average per composition 
    # (only one composition included here)
    av_dens = average_densities([trj1, trj2], tstart=0)
    for comp in av_dens:
        binc = av_dens[comp]['binc']
        hist = av_dens[comp]['hists']
        plot_density('density_average_%s'%comp, binc, hist)
    ###################################################################

