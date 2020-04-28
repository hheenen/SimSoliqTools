#!/usr/bin/env python

import os
from shutil import copyfile
from simsoliq.io import init_mdtraj
    
    
    ##################################################################
    ### this example demonstrates the preparation of single point  ###
    ### clculations with simsoliq, the example utilizes example    ###
    ### input as used for a VASP calculation. Three different      ### 
    ### types of single points are shown:                          ###
    ###     (1) singlepoints of full system                        ###
    ###     (2) singlepoints of system without solvent atoms       ###
    ###     (3) singlepoints of system with solvent only           ###
    ##################################################################


########################################################################
###################arguments for submit scripts ########################
########################################################################
    
######################## arguments for incar ###########################
incar_keys = {"System":"singlepoint", "ISTART":"0", "PREC":"Normal",\
    "ENCUT":"400", "GGA":"RP", "IVDW":"11", "NCORE":"16", "KPAR":"1",\
    "EDIFF":"1e-4", "LREAL":"AUTO", "EDIFFG":"0", "IBRION":"-1", \
    "NSW":"0", "ISMEAR":"0", "SIGMA":"0.1", "NELM":"200", \
    "LWAVE":".FALSE.", "LCHARG":".TRUE.", "LVHAR":".TRUE.",\
    "IDIPOL":"3", "LDIPOL":".TRUE.", "DIPOL":"0.5 0.5 0.5"}
########################################################################
    
    
################### header slurm submit file ###########################
hdr = "#!/bin/bash\n#SBATCH -J job\n#SBATCH -p xeon16\n"
hdr += "#SBATCH -N 1\n#SBATCH --ntasks-per-node=16\n"
hdr += "#SBATCH -t 20:00:00\n#SBATCH -o slurm.%j.out\n"
hdr += "#SBATCH -e err\n\nmodule purge\nmodule load intel\n"
hdr += "export PATH=/home/cat/zwang/bin/cattheory/vasp/5.4.4:$PATH\n\n"
########################################################################

########################################################################
########################################################################
########################################################################



if __name__ == "__main__":
    # Note: to demonstrate the singlepoint preparation a directory
    #       MD_data is created and singlepoints are prepared therein

    #### prep directory ####
    print("create directory `MD_data` for demonstration")
    dat_path = '../tests/data/Pt111_24H2O_x/'
    if not os.path.isdir("MD_data"):
        os.mkdir("MD_data")
    for f in ['vasprun.xml', 'POTCAR', 'KPOINTS']:
        copyfile(dat_path+'/'+f, "MD_data/"+f)
    ########################



    ##################################################################
    ###     (0) Beginning of the general script                    ###
    ##################################################################
    
    # read MD_data
    a = init_mdtraj("MD_data/vasprun.xml", fmat='vasp')

    
    ##################################################################
    ###     (1) singlepoints of full system                        ###
    ##################################################################
    
    # make singlepoints for vasp-WF calculations for every=freq snapshots
    
    spfolders = a.prepare_singlepoint_calculations(tag='WF', freq=1, **incar_keys)

    # use returned list of prepared folders to create bash-script for
    # for chained jobs via simsoliq scripts
    import sys
    sys.path.append("../scripts")
    from submit_tools import write_chain_submit

    
    write_chain_submit("MD_data/singlepoints_WF/submit_chain.sh",
        hdr, "mpirun vasp_std", spfolders, 100)

    
    ##################################################################
    ###     (2) singlepoints of system without solvent atoms       ###
    ##################################################################

    # create solvent free atom structures
    atoms_slab = [a.isolate_nosolvent_snapshot(i) for i in range(a.mdtrajlen)]
    
    # prep sinplepoint folders
    spfolders = a.prepare_singlepoint_calculations(tag='nosolvent', freq=1, \
        spatoms=atoms_slab, **incar_keys)

    # prepare submission script
    write_chain_submit("MD_data/singlepoints_nosolvent/submit_chain.sh",
        hdr, "mpirun vasp_std", spfolders, 100)


    ##################################################################
    ###     (3) singlepoints of system with solvent only           ###
    ##################################################################

    # create solvent only atom structures
    atoms_solv = [a.isolate_solvent_snapshot(i) for i in range(a.mdtrajlen)]
    
    # prep sinplepoint folders
    spfolders = a.prepare_singlepoint_calculations(tag='solvent', freq=1, \
        spatoms=atoms_solv, **incar_keys)

    # prepare submission script
    write_chain_submit("MD_data/singlepoints_solvent/submit_chain.sh",
        hdr, "mpirun vasp_std", spfolders, 100)

