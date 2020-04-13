#!/usr/bin/env python

import numpy as np
from ase.symbols import chemical_symbols
from simsoliq.io import init_mdtraj
from simsoliq.plotting.standard_plots import plot_density
from simsoliq.mdtraj_average import adjust_density_to_substrate

if __name__ == "__main__":
    
    # read trajectory
    a = init_mdtraj("../tests/data/Pt111_24H2O_x/vasprun.xml", \
                    fmat='vasp', concmode='nested', fnest='restart')

    # get density data
    densdata = a.get_density_profile(height_axis=2, savepkl=False)
    binc = densdata['binc']
    hist_dicts = densdata['hists']

    ### plot density of everything in the simulation
    plot_density('density_example_raw', binc, hist_dicts)
    
    ### plot density of solvent
    # (1) truncate data to relevant range by hand
    istart = np.where(hist_dicts['Pt'] > 0.0)[0][-1]+1
    # reduce data to solvent
    hist_solv = {key:hist_dicts[key][istart:] for key in hist_dicts \
        if key.find('solv') != -1}
    
    plot_density('density_solvent_only_1', binc[istart:], hist_solv)

    # (2) use function
    substrate = a.get_substrate_types()[0]
    esub = chemical_symbols[substrate]
    dens2 = adjust_density_to_substrate(densdata, esub)
    plot_density('density_solvent_only_2', dens2['binc'], dens2['hists'])

