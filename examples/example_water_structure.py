#!/usr/bin/env python

import numpy as np
from simsoliq.io import init_mdtraj
from simsoliq.analyze import evaluate_h2o_adsorption, summarize_h2o_adsorption_output, get_slab_sites, visualize_sites


if __name__ == "__main__":
    # load trajectory, of Pt(111) with 24 H2O and *OH adsorbate
    a = init_mdtraj("../tests/data/Pt111_24H2O_OH_long/vasprun.xml")
    print(a)

    # obtained site-tags (only possible for ASE slabs: 100,110,111,211)
    visualize_sites(a, facet='111', slabsize=(3,4), view=True)
    site_data = get_slab_sites(a, facet='111', slabsize=(3,4))
    
    # obtain raw data for h2o adsorption
    h2o_out = evaluate_h2o_adsorption(a, site_data, dchs=2.55)
    # make data simpler
    out = summarize_h2o_adsorption_output(h2o_out, tstart=0)

    print('H2O adsorption:', out)

