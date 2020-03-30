#!/usr/bin/env python

import numpy as np
from simsoliq.io import init_mdtraj
from simsoliq.plotting.standard_plots import plot_engs_vs_time, \
        plot_av_engs_vs_time

if __name__ == "__main__":
    
    # read trajectory
    a = init_mdtraj("../tests/data/Pt111_24H2O_x/vasprun.xml", \
                    fmat='vasp', concmode='nested', fnest='restart')

    # print some information
    print(a)

    # make energy plot
    eng_data = a.get_traj_energies()
    plot_engs_vs_time("energy_vs_time", eng_data, tstart=0.0)
    plot_av_engs_vs_time("average_energy_vs_time", \
        eng_data, tstart=0.0, tunit='fs')

    etot_data = {k:eng_data[k] for k in eng_data if k not in ['ekin','epot']}
    plot_av_engs_vs_time("average_total_energy_vs_time", \
        etot_data, tstart=0.0, tunit='fs')

