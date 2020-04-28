"""
This module contains functionality for creating an mdtraj-object
capable of reading VASP formats

(for later) This module contains functions for reading vasp
input to be inherited by an mdtraj object

"""

import os, subprocess
import numpy as np
from shutil import copyfile
from ase.io import read, write
from ase.build import sort
from ase.calculators import vasp
from simsoliq.mdtraj import MDtraj
from simsoliq.io.utils import _unpack_files, lines_key
from simsoliq.geometry_utils import find_max_empty_space

def read_mdtraj_vasp(fpath, fname, **kwargs):
    """
      function to create a 'vasp-mdtraj' (io) object
      passes arguments to object (see function io.read_mdtraj)

    """

    mdtraj = MDtraj(bpath=fpath, fident=fname, **kwargs)
    mdtraj.efunc = _read_energies_vasp
    mdtraj.afunc = _read_vasp_atoms
    mdtraj.tfunc = _read_vasp_timedata
    mdtraj.sp_prep = _prep_vasp_singlepiont
    mdtraj.efunc_sp = _get_sp_epot
    mdtraj.fefunc_sp = get_sp_efermi
    mdtraj.vacefunc_sp = get_sp_vacpot
    mdtraj.wffunc_sp = get_vasp_wf
    mdtraj.chgfunc_sp = get_charge_density

    return(mdtraj)



############## vasp specific functions to read output files ###################

def _read_energies_vasp(wpath, fident):
    """ 
      function to identify reading data from OUTCAR or vasprun.xml
    
    """
    if fident.find("OUTCAR") != -1:
        return(_read_OUTCAR_energetics(wpath+"/"+fident))
    elif np.all([fident.find(k) != -1 for k in ["vasprun","xml"]]):
        return(_read_xml_energetics(wpath+"/"+fident))


def _read_xml_energetics(filename):
    """
      hard-coded parser for energies in xml files
      TODO: replace with xml parser
    
    """
    process = subprocess.Popen("grep '%s' -B 4 %s"%('"kinetic"',filename),\
                           shell=True,stdout=subprocess.PIPE)
    tstr = process.communicate()[0].decode('utf-8').split()
    #tstr = os.popen("grep '%s' -B 4 %s"%("kinetic",filename)).read().split()
    epot = np.array([float(s) for s in tstr[3::18]])
    ekin = np.array([float(s) for s in tstr[15::18]])
    etot = ekin+epot
    return({'ekin':ekin,'epot':epot,'etot':etot,'runlengths':[etot.size]})


def _read_OUTCAR_energetics(filename):
    """ 
      hard-coded parser for energies in OUTCAR files
      TODO: replace with more reliable code
    
    """
    # Thermodynamics
    ekin = _get_val_OUTCAR(filename,'kinetic energy EKIN   =',4,5)
    epot = _get_val_OUTCAR(filename,'ion-electron   TOTEN  =',4,7)
    etot = ekin+epot
    return({'ekin':ekin,'epot':epot,'etot':etot,'runlengths':[etot.size]})


def _get_val_OUTCAR(filename,string,i,j):
    """ 
      primitive helper function
      TODO: replace with more reliable code
    
    """
    process = subprocess.Popen("grep '%s' %s"%(string,filename),\
                           shell=True,stdout=subprocess.PIPE)
    tstr = process.communicate()[0].decode('utf-8').split()
    
    val = [float(s) for s in tstr[i::j]]
    return(np.array(val))


def _read_vasp_atoms(wpath, fident, safe_asetraj_files=True):
    """
      function to read all structures of a vasp-output file
      as a list of ase-atoms objects
 
      Parameters
      ----------
      wpath : str
          name of path in which output file lies
      fident : str
          name / identifier of file to read
      safe_asetraj_file : bool
          boolean to decide whether simsoliq shall save/use
          trajectory files of output data for each output file
 
      Returns
      -------
      atoms : list of ase-atoms objects
          list of ase-atoms objects of each snapshot of the output file
    
    """
    
    savedfile = wpath+'/mdtraj_atoms_%s.traj'%(fident.split('.')[0].lower())
    
    # no saved file existing
    if not os.path.isfile(savedfile) or \
        os.path.getmtime(savedfile) < os.path.getmtime(wpath+'/'+fident):
        atoms = read(wpath+'/'+fident,':')
        if safe_asetraj_files:
            # save positional data
            write(savedfile, atoms)
    else:
        atoms = read(savedfile,':')
    return(atoms)
    
    
def _read_vasp_timedata(wpath, fident):
    """
      function to read meta data for timestep used in trajectory
 
      Parameters
      ----------
      wpath : str
          name of path in which output file lies
      fident : str
          name / identifier of file to read
 
      Returns
      -------
      timedat : dict
          includes "timeunit" and "timestep"
    
    """
    #TODO: needs proper implementation hard-coded    
    timedat = {"timeunit":['fs'], "timestep":[1]}
    return(timedat)


def _prep_vasp_singlepiont(cpath, bpath, atoms, **kwarks):
    """
      function to prepare a vasp single point calculation 
      based on files in `bpath`; only created if no existing calculation
      
 
      Parameters
      ----------
      cpath : str
          name of path for singlepoint calculation
      bpath : str
          name of path of original AIMD simulation
      atoms : ASE-atoms object
          structure of which to calculate a singlepoint
 
      Returns
      -------
      prep : bool
          boolean indicating if singlepoint was prepared
    
    """
    # only prepare if path does not contain converged singlepoint
    if not _converged_singlepoint(cpath):
        atoms = sort(atoms, tags=atoms.get_atomic_numbers())
        write(cpath+"/POSCAR",atoms)
        grep_PAW_POTCAR(bpath+"/POTCAR", cpath+'/POTCAR', atoms)
        copyfile(bpath+"/KPOINTS",cpath+"/KPOINTS")
        write_incar(cpath+"/INCAR", kwarks) # could be more clever
        return(True)
    else:
        return(False)


def grep_PAW_POTCAR(fpot, fout, atoms):
    """ 
      helper-function to extract PAW potentials for atoms object
    
    """
    # set up element to search PAW for
    el_list = []
    for e in atoms.get_chemical_symbols():
        if len(el_list) == 0 or e != el_list[-1]:
            el_list.append(e)
    
    with open(fpot, 'r') as pfile:
        lines = pfile.readlines()

    # grep lines from POTCAR
    ldat = {}
    for l in range(len(lines)):
        if len(lines[l].split()) > 0 and lines[l].split()[0] == 'PAW_PBE':
            n = 0
            while lines[l+n].find('End of Dataset') == -1:
                n += 1
            ldat.update({lines[l].split()[1]:lines[l:l+n+1]})

    # write lines in OUtFILE
    with open(fout, 'w') as outfile:
        for el in el_list:
            outfile.write(''.join(ldat[el]))


def write_incar(fincar, keys):
    """ 
      helper-function to write INCAR file
    
    """
    incartxt = ''
    for key in keys:
        incartxt += "%s = %s\n"%(key,keys[key])
    with open(fincar, 'w') as outfile:
        outfile.write(incartxt)


def _converged_singlepoint(cpath):
    """ 
      helper-function to check singlepoint convergence
    
    """
    
    # in case its packed
    _unpack_files(cpath,['OUTCAR','OSZICAR'])
    
    if os.path.isfile(cpath+'/OUTCAR') and os.path.isfile(cpath+'/OSZICAR'):
        with open(cpath+'/OUTCAR','r') as f:
            olines = f.readlines()
        with open(cpath+'/OSZICAR','r') as f:
            zlines = f.readlines()
    
        # finished OUTCAR-file ?
        check = True and lines_key(olines,\
            'General timing and accounting informations',\
            0,1,rev=True,loobreak=True) == 'timing'
        
        # last iteration didn't hit NELM ?
        nscflast = int(lines_key(zlines,'DAV:',0,1,rev=True,loobreak=True))
        nelm = 0
        if lines_key(olines,'NELM   =',0,2,rev=False,loobreak=True) != False:
            nelm = int(lines_key(olines,'NELM   =',0,2,rev=False,loobreak=True)[:-1])
        check = check and nscflast < nelm
        
       ## if geo-opt converged?
       ##fconv = float(lines_key(olines,'EDIFFG =',0,2,rev=False,loobreak=True))
       #fs = [float(line.split()[2]) for line in zlines if line.find('F=') != -1]
       #check = check and (lines_key(olines,\
       #    "reached required accuracy - stopping structural energy minimisation",\
       #    0,1,rev=True,loobreak=True) == 'required' or len(fs) == 1)
        return(check)
    else:
        return(False)


def _get_sp_epot(sp_path):
    """ 
      helper-function to return singlepoint epot
    
    """
    if _converged_singlepoint(sp_path):
        with open(sp_path+'/OUTCAR','r') as f:
            olines = f.readlines()
        epot = float(lines_key(olines, 'free  energy   TOTEN  =',0,4,loobreak=True))
        return(epot)
    else:
        return(None)


def get_sp_efermi(sp_path):
    """ 
      helper-function to return singlepoint fermi energy
    
    """
    if _converged_singlepoint(sp_path):
        with open(sp_path+'/OUTCAR','r') as f:
            lines = f.readlines()
        ef = float(lines_key(lines,'E-fermi',0,2,rev=True,loobreak=True))
        return(ef)
    else:
        return(None)


def get_sp_vacpot(sp_path, **kwargs):
    """ 
      helper-function to return singlepoint potential data
      from LOCPOT
    
    """
    _unpack_files(sp_path,['OUTCAR','LOCPOT']) #in case its packed
    if not os.path.isfile(sp_path+'/LOCPOT'):
        raise IOError("no LOCPOT located in %s"%sp_path)

    # read and transform LOCPOT
    d0, atoms = _read_potential_vasp(sp_path)
    ind_vac = _determine_vacuum_reference(d0, atoms, **kwargs)

    # by default take refernce `before` dipole correction (higher z-coordinate)
    ind_vac = ind_vac[0]

    ref_vac = d0[ind_vac]
    return(ref_vac)


def get_vasp_wf(sp_path, **kwargs):
    """ 
      helper-function to return singlepoint workfunction
      from LOCPOT and OUTCAR
    
    """

    ef = get_sp_efermi(sp_path)
    evac = get_sp_vacpot(sp_path, **kwargs)
    
    wf = evac - ef
    return(wf)


def _read_potential_vasp(f):
    """ 
      helper-function to return electrostatic potential from OUTCAR and LOCPOT
      hardcoded to z-axis
    
    """
    # use ASE to read OUTCAR for volumetric data
    atoms = read(f+'/OUTCAR')
    
    # local potential
    c=vasp.VaspChargeDensity(filename=f+'/LOCPOT')
    c.read(filename=f+'/LOCPOT')
    # normalize to volume
    dens=c.chg[0]*atoms.get_volume() # ASE factors wronlgy via volume
    d0=np.mean(np.mean(dens,axis=0),axis=0)
    return(d0, atoms)


def _determine_vacuum_reference(d0, atoms, gtol=1e-3):
    """ 
      helper-function to automatically obtain vacuum reference
      hardcoded to z-axis, theoratically usable independent of code
    
    """
    # gtol == tolerance for when potential is 'converged'/flat in V/AA
    # vacuum region - first guess
    z=np.linspace(0,atoms.cell[2,2],len(d0))
    z_vac = find_max_empty_space(atoms,edir=3)
    ind_vac = np.absolute(z - z_vac).argmin()

    # double potential for simpler analysis at PBC
    d02 = np.hstack((d0,d0))
    # investigate at gradient
    gd02 = np.absolute(np.gradient(d02))
    iv2 = ind_vac+d0.size
    # search max gradient as center of dipole correction withint 3 AA
    diA = np.where(z > 1.0)[0][0]
    imax = iv2-diA*2 + gd02[iv2-diA*2:iv2+diA*2].argmax()

    # walk from imax to find minimum
    # ibfe/iafr --> before/after dipole correction along z-axis
    ibfe = np.where(gd02[imax-diA*3:imax] < gtol)[0] + imax-diA*3
    iafr = np.where(gd02[imax:imax+diA*3] < gtol)[0] + imax

    if ibfe.size == 0 or iafr.size == 0:
        print('###\nproblematic WF gradient\n###')
        ibfe = iafr = [ind_vac]
    ibfe = ibfe[-1]%d0.size; iafr = iafr[0]%d0.size
    
    return([ibfe, iafr])


def get_charge_density(f):
    """ 
      helper-function to read a CHGCAR and return its electron density
    
    """
    # read CHGCAR
    _unpack_files(f,['CHGCAR']) #in case its packed
    filename = f+'/CHGCAR'
    if not os.path.isfile(filename):
        raise IOError("no CHGCAR found in %s"%f)

    chg = _read_vasp_chgcar_density(filename)

    return(chg)


def _read_vasp_chgcar_density(filename):
    """
    Helper function to read CHGCAR 
    
    """
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    
    #get unit cell volume
    a = np.array([float(lines[2].split()[i]) for i in range(3)])
    b = np.array([float(lines[3].split()[i]) for i in range(3)])
    c = np.array([float(lines[4].split()[i]) for i in range(3)])
    V = np.dot(np.cross(b,c),a)
    
    #get header length and find nx, ny, nz
    for i,line in enumerate(lines):
        if len(line.strip()) == 0:
            break
    i += 1
    line = lines[i]
    vox_n = np.array([int(line.split()[j]) for j in range(3)])
    n_tot = vox_n[0]*vox_n[1]*vox_n[2]
    i += 1
    
    #find footer length
    for k in range(len(lines)):
        if lines[-k].split()[0] == 'augmentation' and lines[-k].split()[2] == '1':
            break
    
    #and return the xy-averaged electron density at the specified z value
    density = np.genfromtxt(filename,skip_header=i,skip_footer=k,invalid_raise=False)
    density = np.reshape(density,density.size)
    if density.size < n_tot:
        for chg in lines[-k-1].split():
            density = np.append(density,float(chg))
    density = np.reshape(density,vox_n,order='F')
    density /= n_tot
    
    return density


