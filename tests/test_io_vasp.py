import unittest

import numpy as np
from simsoliq.io import init_mdtraj


# data for trajectory x
ekin = np.array([3.393914, 3.453727, 3.454822, 
                3.351518, 3.242996, 3.320347])
epot = np.array([-558.760232, -558.772116, -558.739619, 
                -558.593003, -558.43692, -558.488026])
etot = np.array([-555.366318, -555.318389, -555.284797, 
                 -555.241485, -555.193924, -555.167679])

def compare_arrays(earr0, earr1, prec=5):
    return(np.array_equal(np.around(earr0,prec), np.around(earr1,prec)))

class TestIOVasp(unittest.TestCase):

    def test_get_energies_vasprun(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode=None)
        self._eval_engs(a)

    def test_get_energies_outcar(self):
        a = init_mdtraj("data/Pt111_24H2O_x/OUTCAR", fmat='vasp')
        self._eval_engs(a)
    
    def test_get_energies_nested(self):
        a = init_mdtraj("data/Pt111_24H2O_x/OUTCAR", fmat='vasp', concmode='nested', fnest='restart')
        self._eval_engs(a)
    
    def test_get_energies_level(self):
        a = init_mdtraj("data/Pt111_24H2O_x/OUTCAR*", fmat='vasp', concmode='level')
        self._eval_engs(a)
        
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun*.xml", fmat='vasp', concmode='level')
        self._eval_engs(a)
    
    def _eval_engs(self, a):
        # test dictionary
        engs = a.get_traj_energies()
        ntraj = a.mdtrajlen
        self.assertTrue(compare_arrays(ekin[:ntraj], engs['ekin']))
        self.assertTrue(compare_arrays(epot[:ntraj], engs['epot']))
        self.assertTrue(compare_arrays(etot[:ntraj], engs['etot']))

        # test individual functions
        self.assertTrue(compare_arrays(ekin[:ntraj], a.get_kinetic_energies()))
        self.assertTrue(compare_arrays(epot[:ntraj], a.get_potential_energies()))
        self.assertTrue(compare_arrays(etot[:ntraj], a.get_total_energies()))
        
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


