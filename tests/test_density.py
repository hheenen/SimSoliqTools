import unittest

import numpy as np
from simsoliq.io import init_mdtraj


class TestDensity(unittest.TestCase):

    def test_get_solvent_indices(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode='nested', fnest='restart')
        sident = a._get_solvent_indices(snapshot=0, smol=[8,2], rcut=1.3)

        sident_compare = {48: [0, 1], 49: [2, 3], 50: [4, 5], 51: [6, 7], 52: [8, 9], 
            53: [10, 11], 54: [12, 13], 55: [14, 15], 56: [17, 40], 57: [18, 19], 
            58: [20, 21], 59: [22, 23], 60: [24, 25], 61: [26, 27], 62: [28, 29], 
            63: [30, 31], 64: [32, 33], 65: [34, 35], 66: [36, 37], 67: [38, 39], 
            68: [16, 41], 69: [42, 43], 70: [44, 45], 71: [46, 47]}

        for entry in sident:
            self.assertIn(entry,sident_compare)
            self.assertTrue(np.array_equal(sident[entry],sident_compare[entry]))
    
    def test_get_density(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode='nested', fnest='restart')
        densdata = a.get_density_profile(height_axis=2, savepkl=False)
        
        # only test if same number of `populated bins`
        out_compare = {'Osolv':23, 'Hsolv':37, 'Pt':5}

        binc, hist_dicts = densdata['binc'], densdata['hists']
        for key in hist_dicts:
            self.assertEqual(np.where(hist_dicts[key] != 0.0)[0].size, out_compare[key])
