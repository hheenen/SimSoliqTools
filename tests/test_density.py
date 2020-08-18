import unittest

import numpy as np
from simsoliq.io import init_mdtraj
from simsoliq.mdtraj_average import average_densities
from simsoliq.analyze.density import isolate_solvent_density, get_peak_integral, get_average_solvent_bulk_density


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

    def test_average_density(self):
        trj1 = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp')
        trj2 = init_mdtraj("data/Pt111_24H2O_x/restart/vasprun.xml", fmat='vasp')
        av_dens = average_densities([trj1, trj2], tstart=0)

        # compare number of `populated bins`
        out_compare = {'Osolv':23, 'Hsolv':37, 'Pt':5}
        hists = av_dens['Pt36_24H2O']['hists']
        for key in hists:
            self.assertEqual(np.where(hists[key] != 0.0)[0].size, out_compare[key])

    def test_analyze_density(self):
        trj = init_mdtraj("../tests/data/Pt111_24H2O_OH_long/vasprun.xml")
        densdata = trj.get_density_profile(height_axis=2, savepkl=True)

        # truncation of density
        dens = isolate_solvent_density(densdata)

        # integrals of first peaks
        dint_m = get_peak_integral(dens, xlim=None) # minimum after first maximum
        self.assertEqual(np.around(dint_m['Osolv'][1],12), 0.000000001541)
        
        Acell = trj.get_cell_area() # required cell area
        dint_n = get_peak_integral(dens, xlim=None, Acell=Acell)
        self.assertEqual(np.around(dint_n['Osolv'][1],5), 7.46394)

if __name__ == '__main__':
    unittest.main()


