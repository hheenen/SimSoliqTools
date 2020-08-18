#!/usr/bin/env python

import numpy as np
from simsoliq.io import init_mdtraj
from simsoliq.analyze.water_structure import partition_solvent_composition, evaluate_h2o_adsorption, summarize_h2o_adsorption_output
from simsoliq.analyze.sampling_sites import get_slab_sites, visualize_sites, get_top_slab_atoms


if __name__ == "__main__":
    # load trajectory, of Pt(111) with 24 H2O and *OH adsorbate
    a = init_mdtraj("../tests/data/Pt111_24H2O_OH_long/vasprun.xml")
    print(a)

  # # short example only of basic functionality and output
  # sollist = partition_solvent_composition(a)
  # sub_inds = a._get_substrate_indices()
  # natoms = len(a.get_single_snapshot(n=0)); nsolv = len(sollist[0]['solv_inds']); nsub = len(sub_inds)
  # print(natoms-nsolv-nsub, natoms, nsolv, nsub)
  # assert False
    # TODO: try a fool-proof stoichiometry averaging (clear identification of adsorbates)

    # obtained site-tags (only possible for ASE slabs: 100,110,111,211)
    site_data = get_slab_sites(a, facet='111', slabsize=(3,4))
    
    # obtain raw data for h2o adsorption
    h2o_out = evaluate_h2o_adsorption(a, site_data, dchs=2.55)
    # make data simpler
    out = summarize_h2o_adsorption_output(h2o_out, tstart=0)

    print('H2O adsorption:', out)

