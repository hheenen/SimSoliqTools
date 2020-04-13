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
    

if __name__ == '__main__':
    unittest.main()
