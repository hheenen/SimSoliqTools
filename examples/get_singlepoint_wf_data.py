#!/usr/bin/env python

from simsoliq.io import init_mdtraj

def print_singlepoint_data(key, data):
    outtxt = 5*"#" + " singlepoint data: {:>5} ".format(key) + 5*"#" + "\n"
    tkeys = list(data.keys()); tkeys.sort()
    for t in tkeys:
        outtxt += 'ts {:05d} = {:}\n'.format(t,data[t])
    outtxt += 35*"#"
    print(outtxt)
    

if __name__ == "__main__":
    # read MD_data path
    a = init_mdtraj("../tests/data/Pt111_24H2O_x/vasprun.xml", \
                    fmat='vasp', concmode='nested', fnest='restart')
    
    # get info about mdtraj object
    print(a)
    
    # read singlepoint data: potential energy
    sp_epot = a.read_singlepoint_calculations('epot')
    print_singlepoint_data('epot',sp_epot)
    
    # read singlepoint data: fermi energy
    sp_ef = a.read_singlepoint_calculations('efermi')
    print_singlepoint_data('efermi',sp_ef)
    
    # read singlepoint data: vacuum potential
    sp_evac = a.read_singlepoint_calculations('evac')
    print_singlepoint_data('evac',sp_evac)
    
    # read singlepoint data: workfunction
    sp_wf = a.read_singlepoint_calculations('wf')
    print_singlepoint_data('wf',sp_wf)

