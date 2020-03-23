#!/usr/bin/env python

from simsoliq.io import init_mdtraj
import numpy as np

if __name__ == "__main__":
    #a = init_mdtraj("data/Pt111_24H2O_x/OUTCAR")#vasprun.xml")
    a = init_mdtraj("data/Pt111_24H2O_x/OUTCAR", fmat='vasp', concmode='nested', fnest='restart')
    print(a.get_kinetic_energies())
    print(a.get_potential_energies())
    print(a.get_total_energies())

    pos = a.get_traj_atoms()
    print(pos[0].get_cell())
    np.savetxt('data/Pt111_24H2O_x/pos_snap1.txt', pos[1].get_positions())
    print(a)
