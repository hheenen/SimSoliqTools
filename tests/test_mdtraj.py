import unittest

import numpy as np
from simsoliq.io import init_mdtraj
from simsoliq.mdtraj_average import average_energies, sort_energies


class TestMDTraj(unittest.TestCase):
    
    def test_composition(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode=None)
        self.assertEqual(a.get_traj_composition(), "Pt36_24H2O")

    def test_substrate(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode=None)
        self.assertTrue(np.array_equal([78], a.get_substrate_types()))
    
    def test_energy_averaging(self):
        trj1 = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp')
        trj2 = init_mdtraj("data/Pt111_24H2O_x/restart/vasprun.xml", fmat='vasp')

        # mean and std
        dat_compare = {'mean': -555.262098635, 'std': 0.043182517330844446}
        e_tot = average_energies([trj1, trj2], tstart=0)['etot']['Pt36_24H2O']
        for m in e_tot:
            self.assertEqual(e_tot[m], dat_compare[m])

        # data_retrieval and structure
        dat_compare = np.array([[-555.36631751, -555.31838954, -555.28479676],[-555.24148481, -555.19392417, -555.16767902]])
        e_tot = sort_energies([trj1, trj2])['etot']['Pt36_24H2O']
        self.assertTrue(np.array_equal(dat_compare, np.around(e_tot, 8)))

    def test_atom_manipulation(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode='nested', fnest='restart')
       
        ref_data = [{1: 48, 8: 24, 78: 36}, {1: 48, 8: 24}, {78: 36}]

        traj0 = a.get_single_snapshot(n=0)
        solv0 = a.isolate_solvent_snapshot(0)
        slab0 = a.isolate_nosolvent_snapshot(0)

        slabs = [traj0, solv0, slab0]

        for i in range(len(slabs)):
            for k in np.unique(slabs[i].get_atomic_numbers()):
                self.assertEqual(np.where(slabs[i].get_atomic_numbers() == k)[0].size, ref_data[i][k])

    def test_get_time_average_solvent_indices(self):
        a = init_mdtraj("data/Pt111_24H2O_OH_long/vasprun.xml", fmat='vasp')
        ma_ind = a._get_time_average_solvent_indices(savepkl=False)
        c_ma_ind = np.loadtxt("data/Pt111_24H2O_OH_long/solvent_indices.txt")
        self.assertTrue(np.array_equal(ma_ind, c_ma_ind))


if __name__ == '__main__':
    unittest.main()
