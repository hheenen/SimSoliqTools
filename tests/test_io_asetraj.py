import unittest

import numpy as np
from simsoliq.io import init_mdtraj


# data for trajectory x
epot = np.array([-558.56152941, -558.57337521, -558.54074071])

def compare_arrays(earr0, earr1, prec=5):
    return(np.array_equal(np.around(earr0,prec), np.around(earr1,prec)))

class TestIOAsetraj(unittest.TestCase):

    def test_get_energies_vasprun(self):
        a = init_mdtraj("data/Pt111_24H2O_x/mdtraj_atoms_vasprun.traj", fmat='ase', concmode=None)
        self._eval_engs(a)

    def _eval_engs(self, a):
        # test dictionary
        engs = a.get_traj_energies()
        ntraj = a.mdtrajlen
        self.assertTrue(compare_arrays(epot[:ntraj], engs['epot']))
        self.assertTrue(compare_arrays(epot[:ntraj], a.get_potential_energies()))
        
    def test_get_positions(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode='nested', fnest='restart')
        mdtraj = a.get_traj_atoms()

        # compare cell
        cell0 = np.array([[8.34588864, 0., 0.], [0., 9.63700211, 0.], [0., 0., 32.54292636]])
        self.assertTrue(compare_arrays(cell0, mdtraj[0].get_cell()))
        
        # compare positions snapshot 1
        pos0 = np.loadtxt('data/Pt111_24H2O_x/pos_snap1.txt')
        self.assertTrue(compare_arrays(pos0, mdtraj[1].get_positions(), prec=2))

    def test_initialize_timestep(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode='nested', fnest='restart')
        a._initialize_timestep()
        
        self.assertEqual(a.timeunit, 'fs')
        self.assertEqual(a.timestep, 1)


if __name__ == '__main__':
    unittest.main()


