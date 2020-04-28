import unittest

import numpy as np
from simsoliq.io import init_mdtraj


class TestSinglepoints(unittest.TestCase):

    def test_get_data(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode='nested', fnest='restart')
        
        epot_compare = {0:-501.4221, 2:-553.6006, 4:-553.8772}
        sp_epot = a.read_singlepoint_calculations('epot', safe_pkl_files=False)
        for k in sp_epot:
            self.assertEqual(epot_compare[k], np.around(sp_epot[k],4))

        efermi_compare = {0:-1.0076, 2:-0.5044, 4:-0.4707}
        sp_fermi = a.read_singlepoint_calculations('efermi', safe_pkl_files=False)
        for k in sp_fermi:
            self.assertEqual(efermi_compare[k], np.around(sp_fermi[k],4))

    def test_readout_evac(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode='nested', fnest='restart')
        evac = a.vacefunc_sp("data/Pt111_24H2O_x/singlepoints_wf/sp_step00000")
        self.assertEqual(np.around(evac, 5), np.around(4.2471160283258795, 5))

    def test_readout_charge_density(self):
        a = init_mdtraj("data/Pt111_24H2O_x/vasprun.xml", fmat='vasp', concmode='nested', fnest='restart')
        dchg = a.chgfunc_sp("data/Pt111_24H2O_x/singlepoints_wf/sp_step00000")
        self.assertEqual(np.around(dchg.sum(),13),552.0000139832506)


