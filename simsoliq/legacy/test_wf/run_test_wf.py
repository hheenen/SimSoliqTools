#!/usr/bin/env python

import sys, os
sys.path.append("/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/scripts_vasp")
from post_process_traj import _read_potential_vasp, _determine_vacuum_reference, get_efermi

def get_vasp_wf(f, **kwargs): #NOTE: copied from post_process_traj
    #_unpack_files(f,['OUTCAR','LOCPOT']) #in case its packed

    d0, a = _read_potential_vasp(f)
    ind_vac = _determine_vacuum_reference(d0, a, **kwargs)
    
    # fermi energy
    ef = float(get_efermi(f+"/OUTCAR"))
    
    wf = d0[ind_vac] - ef
    return(wf)

if __name__ == "__main__":
    test_folder = "data/Li_Co111_4x4x4_%i"

    for i in range(1,4):
        wf = get_vasp_wf(test_folder%i, viz=True, gtol=1e-2)
        print(wf)
        os.rename('output/potential.pdf', 'output/potential_%i.pdf'%i)

