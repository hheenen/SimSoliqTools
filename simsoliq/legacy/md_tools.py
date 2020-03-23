#!/usr/bin/env python

import numpy as np
from ase.atoms import Atoms

class set_count(object):
    def __init__(self):
        self.container = {}

    def __len__(self):
        return(len(self.container.keys()))

    def add(self,a):
        if (a not in self.container):
            self.container.update({a:0})
        self.container[a] += 1

    def get_count(self):
        countables = self.container.keys()
        counts = np.array([self.container[a] for a in countables])
        return(countables, counts)
        
class atomic_diffusion_tools(object):
    ''' Collection of functions to help 
        post-processing MD simulations for bulk systems
    '''
    def __init__(self):
        pass

    def _isolate_elements(self,atoms,atype):
        ''' function to return a new list of atoms objects only 
            including atomic information of one selected atom type 
            of the original atoms object list
            input:  atoms     = original list of atoms object
                    atype     = atomic type (number) to be isolated
            output: new_atoms = new list of atoms object with only 
                                selected atom type information
        '''
        new_atoms = []
        for i in range(0,len(atoms)):
            ind = np.where(atoms[i].get_atomic_numbers() == atype)[0]
            new_atoms.append(Atoms(numbers = atoms[i].get_atomic_numbers()[ind],
                        positions=atoms[i].get_positions()[ind,:],cell=atoms[i].get_cell(),
                        pbc=True))
        return(new_atoms)
    
    def _get_max_displacement(self,atoms):
        ''' parses displacements from a atoms trajectory (list),
            PBC are accounted for if crossings are reasonably captured
            (translations larger than half the box length will lead to errors)
            input:   atoms    = list of atoms object
            output:  displ    = list of np.arrays containing the displacement
                                vector for each atoms object (accounting for PBC)
                     dpos_max = array giving the maximum displacement of each
                                atom (accounting for PBC)
        '''
        #parse final displacement 
        pos0, pos_b = atoms[0].get_scaled_positions(), atoms[0].get_scaled_positions()
        dpos_max = np.zeros((pos0[:,0].size))
        vec_t = np.zeros((pos0[:,0].size,3))
        displ = []
        for i in range(0,len(atoms)):
            # calculate scaled displacement 
            pos_c = atoms[i].get_scaled_positions()
            vec_c = self._correct_vec(pos_c-pos_b)
            vec_t += vec_c
            pos_b = pos_c
            # scan maximum displacement
            vec_tr = np.dot(vec_t, atoms[i].get_cell())
            dpos = np.linalg.norm((vec_tr),axis=1)
            dpos_max = (np.transpose(np.vstack((dpos_max,dpos)))).max(axis=1)
            # accumulate displacement
            displ.append(vec_tr)
        return(displ,dpos_max)


    def _correct_vec(self,vec):
        ''' correct vectors in fractional coordinates 
            (assuming vectors minimal connection between 2 points)
        '''
        vec[np.where(vec >= 0.5)] -= 1.0
        vec[np.where(vec < -0.5)] += 1.0
        return(vec)
    
    def _calc_current_einstein_interval(self,atoms,charge_dict,interval):
        ''' calculate the Einstein formulation of the current correlation averaged 
            over a lag time of selected atom types from a list of atoms objects
            input:  atoms = list of atoms objects
                  charges = charges in a dictionary for use
                 interval = interval between datapionts taken for sample traj (=number of trajs)
            output: msd   = dictionary for all atom-types
                            np.array of dimensions (len(atoms),4)
                            containing x,y,z and norm(xyz) of the type-msd 
                            for each atoms object
        '''
        from time import time

        #alter traj length to make fit
        tlen = len(atoms) - len(atoms)%interval
        if tlen%interval != 0:
            raise Exception("%i vs %i : bad interval/ntraj - multiple of trajectory length"%(interval,len(atoms)))
        length = len(range(0,tlen,interval))
        
        inds, qmsd, charges = {'full':range(len(atoms[0]))}, {'full':np.zeros((interval,length,4))}, atoms[0].get_atomic_numbers()
        ## add only LiCl ##
        inds.update({'licl':np.array(np.where(atoms[0].get_atomic_numbers() == 3)[0].tolist() +\
                                    np.where(atoms[0].get_atomic_numbers() == 17)[0].tolist())})
        qmsd.update({'licl':np.zeros((interval,length,4))})
        qmsd_corr = {'3_licl_eff':np.zeros((interval,length)),'17_licl_eff':np.zeros((interval,length))}
        ###################
        charges = np.array([charge_dict[el] for el in charges])

        atypes = np.unique(atoms[0].get_atomic_numbers())
        for atype in atypes:
            qmsd.update({atype:np.zeros((interval,length,4))})
            inds.update({atype:np.where(atoms[0].get_atomic_numbers() == atype)[0]})
        
        displ_tot, dpos_max = self._get_max_displacement(atoms) #complete vec hist
        for i in range(0,interval):
            t0 = time()
            displ = displ_tot[i:tlen:interval] - displ_tot[i] #simple and works
            step_save = {}
            for dtyp in qmsd:
                tqd = self._displ2qd(displ,inds[dtyp],charges)
                qmsd[dtyp][i][:,:] = tqd**2.0
                step_save.update({dtyp:tqd}) #save for correlated motion calc
            
            # calculate effective ion motion - for li and cl
            qmsd_li_eff, qmsd_cl_eff = self._calc_eff_qmsd(step_save[3],step_save[17])
            qmsd_corr['3_licl_eff'][i][:] = qmsd_li_eff
            qmsd_corr['17_licl_eff'][i][:] = qmsd_cl_eff
            
            if (i%200 == 0):
                print("at %i of %i with %.2f sec per calc"%(i,interval,time()-t0))
        qmsd.update(qmsd_corr)
        return(qmsd)

    def _calc_current_einstein_time_averaged(self,atoms,charge_dict,lag,extra_inds={}):
        ''' calculate the Einstein formulation of the current correlation averaged 
            over a lag time of selected atom types from a list of atoms objects
            input:  atoms = list of atoms objects
                  charges = charges in a dictionary for use
                    lag   = ratio of trajectory length giving lagtime over which 
                            to average
               extra_inds = indices of atoms to be singled out for current calculation
            output: msd   = dictionary for all atom-types
                            np.array of dimensions (len(atoms),4)
                            containing x,y,z and norm(xyz) of the type-msd 
                            for each atoms object
        '''
        from time import time
        length = int(len(atoms)*lag) #images using for averaging

        inds, qmsd, charges = {'full':range(len(atoms[0]))}, {'full':np.zeros((length,4))}, atoms[0].get_atomic_numbers()
        qd = {'full':np.zeros((length,4))}
        ## add only LiCl ##
        inds.update({'licl':np.array(np.where(atoms[0].get_atomic_numbers() == 3)[0].tolist() +\
                                    np.where(atoms[0].get_atomic_numbers() == 17)[0].tolist())})
        qmsd.update({'licl':np.zeros((length,4))})
        qd.update({'licl':np.zeros((length,4))})
        qmsd_corr = {'3_licl_eff':np.zeros((length)),'17_licl_eff':np.zeros((length))}
        ## add extra inds TODO: solve (most) issues this way as extra keywords
        [qmsd.update({key_val[0]:np.zeros((length,4))}) for key_val in extra_inds.items()]
        [qd.update({key_val[0]:np.zeros((length,4))}) for key_val in extra_inds.items()]
        [inds.update({key_val[0]:key_val[1]}) for key_val in extra_inds.items()]
        ###################
        charges = np.array([charge_dict[el] for el in charges])

        atypes = np.unique(atoms[0].get_atomic_numbers())
        for atype in atypes:
            qmsd.update({atype:np.zeros((length,4))})
            qd.update({atype:np.zeros((length,4))})
            inds.update({atype:np.where(atoms[0].get_atomic_numbers() == atype)[0]})
       
        for i in range(0,len(atoms)+1-length):
            t0 = time()
            displ, dpos_max = self._get_max_displacement(atoms[i:i+length])
            step_save = {}
            for dtyp in qmsd:
               #qmsd[dtyp] += self.__displ2qmsd(displ,inds[dtyp],charges)
                tqd = self._displ2qd(displ,inds[dtyp],charges)
                qd[dtyp] += tqd
                qmsd[dtyp] += tqd**2.0
                step_save.update({dtyp:tqd}) #save for correlated motion calc
            
            # calculate effective ion motion - for li and cl
            qmsd_li_eff, qmsd_cl_eff = self._calc_eff_qmsd(step_save[3],step_save[17])
            qmsd_corr['3_licl_eff'] += qmsd_li_eff
            qmsd_corr['17_licl_eff'] += qmsd_cl_eff
            
            if (i%100 == 0):
                print("at %i of %i with %.2f sec per calc"%(i,len(atoms)-length,time()-t0))
        qmsd.update(qmsd_corr)

        for dtyp in qmsd:
            qmsd[dtyp] /= len(range(0,len(atoms)+1-length))
            if dtyp in qd:
                qd[dtyp] /= len(range(0,len(atoms)+1-length))
        return(qd, qmsd)

    def _calc_eff_angle(self,qd1,qd2):
        ''' qd1, qd2 are arrays [timestep,qd] calculate angle between
        '''
        scalar = qd1[:,0]*qd2[:,0] + qd1[:,1]*qd2[:,1] + qd1[:,2]*qd2[:,2]
        r_qd1_norm = 1./np.linalg.norm(qd1[:,:3],axis=1)
        r_qd2_norm = 1./np.linalg.norm(qd2[:,:3],axis=1)
        scalar = scalar * r_qd1_norm * r_qd2_norm
        angle = np.arccos(scalar)
        return(angle)
    
    def _calc_eff_qmsd(self,qd1,qd2):
        ''' qd1, qd2 are arrays [timestep,qd] calculate effective transport between them 
            cosine rule based correlation (+ sign due to q*factored msd)
        '''
        qmsd1_eff, qmsd2_eff = np.zeros(qd1[:,0].size), np.zeros(qd2[:,0].size)
        corr = qd1[:,0]*qd2[:,0] + qd1[:,1]*qd2[:,1] + qd1[:,2]*qd2[:,2]
        
        qmsd1_eff = np.linalg.norm(qd1[:,:3], axis=1)**2.0 + corr
        qmsd2_eff = np.linalg.norm(qd2[:,:3], axis=1)**2.0 + corr
        return(qmsd1_eff, qmsd2_eff)

    def _displ2qd(self,displ,ind,charges):
        ''' inner function for msd calculation:
            input:  displ = list of per ion xyz-displacements for a 
                            series of snapshots
                    ind   = indices of atoms for which to calculate the qmsd
                  charges = dictionary for the charges to use in the qmsd
            output: qmsd   = Einstein formulation of the current correlation
                             np.array(len(atoms),4)
        '''
        qd = []
        for i in range(0,len(displ)):
            q_x = np.sum(charges[ind]*displ[i][ind,0])
            q_y = np.sum(charges[ind]*displ[i][ind,1])
            q_z = np.sum(charges[ind]*displ[i][ind,2])
            q_r = np.linalg.norm(np.array([q_x,q_y,q_z]))

            qd.append([q_x, q_y, q_z, q_r])
        return(np.array(qd))
    
    def _calc_msd_interval(self,atoms,interval):
        ''' calculate the Einstein formulation of the current correlation averaged 
            over a lag time of selected atom types from a list of atoms objects
            input:  atoms = list of atoms objects
                 interval = interval between datapionts taken for sample traj (=number of trajs)
            output: msd   = dictionary for all atom-types
                            np.array of dimensions (len(atoms),4)
                            containing x,y,z and norm(xyz) of the type-msd 
                            for each atoms object
        '''
        from time import time
        
        tlen = len(atoms) - len(atoms)%interval
        if tlen%interval != 0:
            raise Exception("%i vs %i : bad interval/ntraj - multiple of trajectory length"%(interval,len(atoms)))
        length = len(range(0,tlen,interval))
       
        atypes=[3,8,17]
        msd, inds = {}, {}
        for atype in atypes:
            msd.update({atype:np.zeros((interval,length,4))})
            inds.update({atype:np.where(atoms[0].get_atomic_numbers() == atype)[0]})
        
        displ_tot, dpos_max = self._get_max_displacement(atoms) #complete vec hist
        for i in range(0,interval):
            t0 = time()
            displ = displ_tot[i:tlen:interval] - displ_tot[i] #simple and works
            for atype in atypes:
                msd[atype][i][:,:] = self._displ2msd(displ,inds[atype])
            if (i%200 == 0):
                print("at %i of %i with %.2f sec per calc"%(i,interval,time()-t0))
        return(msd)

    def _calc_msd_time_averaged(self,atoms,inds={},atypes={},lag=1.0):
        ''' calculate mean square displacement (msd) averaged over
            a lag time of selected atom types from a list of atoms objects
            input:  atoms = list of atoms objects
                    atype = atom type for which to compute msd
                    lag   = ratio of trajectory length giving lagtime over which 
                            to average (1.0 = no lag)
            output: msd   = dictionary for all atom-types
                            np.array of dimensions (len(atoms),4)
                            containing x,y,z and norm(xyz) of the type-msd 
                            for each atoms object
        '''
        from time import time
        length = int(len(atoms)*lag) #images using for averaging
        if len(inds) == 0 and len(atypes) != 0:
            for atype in atypes:
                inds = {atype:np.where(atoms[0].get_atomic_numbers() == atype)[0] for atype in atypes}
        msd = {}
        for typ in inds:
            msd.update({typ:np.zeros((length,7))})
        for i in range(0,len(atoms)+1-length):
            t0 = time()
            displ, dpos_max = self._get_max_displacement(atoms[i:i+length])
            for atype in inds:
                msd[atype] += self._displ2msd(displ,inds[atype])
            if (i%100 == 0):
                print("at %i of %i with %.2f sec per calc"%(i,len(atoms)-length,time()-t0))
        for atype in inds:
            msd[atype] /= len(range(0,len(atoms)+1-length))
        return(msd)

    def _displ2msd(self,displ,ind):
        ''' inner function for msd calculation:
            input:  displ = list of per ion xyz-displacements for a 
                            series of snapshots
                    ind   = indices of atoms for which to calculate the msd
            output: msd   = mean squared displacement of dimensions (len(atoms),4)
        '''
        msd = []
        for i in range(0,len(displ)):
            msd_x = np.mean(displ[i][ind,0]*displ[i][ind,0])
            msd_y = np.mean(displ[i][ind,1]*displ[i][ind,1])
            msd_z = np.mean(displ[i][ind,2]*displ[i][ind,2])
            r_xy = np.linalg.norm(displ[i][ind,:2],axis=1)
            r_xz = np.linalg.norm(displ[i][ind,::2],axis=1)
            r_yz = np.linalg.norm(displ[i][ind,1:],axis=1)
            r_xyz = np.linalg.norm(displ[i][ind,:],axis=1)
            msd_xy = np.mean(r_xy*r_xy)
            msd_xz = np.mean(r_xz*r_xz)
            msd_yz = np.mean(r_yz*r_yz)
            msd_xyz = np.mean(r_xyz*r_xyz)
            msd.append([msd_x,msd_y,msd_z,msd_xy,msd_xz,msd_yz,msd_xyz])
        return(np.array(msd))
    
    def _calc_msd(self,atoms,atype):
        ''' calculates the mean square displacement (msd) of a selected 
            atom type from a list of atoms objects 
            input:  atoms = list of atoms objects
                    atype = atom type for which to compute msd
            output: msd   = np.array of dimensions (len(atoms),4)
                            containing x,y,z and norm(xyz) of the type-msd 
                            for each atoms object
        '''
        ind = range(atoms[0].get_atomic_numbers().size)
        if (atype != None):
            ind = np.where(atoms[0].get_atomic_numbers() == atype)[0] #isolate type
        displ, dpos_max = self._get_max_displacement(atoms)
        msd = self._displ2msd(displ,ind)
        return(msd)

    def _accumulate_type_atoms_displ(self,atoms,atype,dmin=3.):
        ''' creates a new atoms object from list of atoms objects accumulating
            atom positions for chosen types with minimal displacement dmin
            - this is usefull to visualize where atom migration occurred
            input:  atoms = list of atoms objects
                    atype = atom type selected
                    dmin  = minimal displacement for selection
            output: dens  = atoms object including all positions of atoms
                            fullfilling atype and dmin criteria
        '''
        ind = np.where(atoms[0].get_atomic_numbers() == atype)[0] #isolate type
        displ, d_max = self._get_max_displacement(atoms)
        ind2 = ind[np.where(d_max[ind] > dmin)[0]] #isolate type-displ
        dens = self._accumulate_atoms_pos(atoms,ind2)
        return(dens)

    def _accumulate_atoms_pos(self,atoms,ind):
        ''' creates a new atoms object from list of atoms objects accumulating
            atom positions via given indices
            input:  atoms = list of atoms objects
                    ind   = selected atom indices
            output: dens  = atoms object including all positions of selected atoms
        '''
        pos_d, atno = np.zeros((ind.size*len(atoms),3)), np.zeros((ind.size*len(atoms)))
        # accumulate positions of isolated atoms
        for i in range(0,len(atoms)):
            pos = atoms[i].get_positions()
            pos_d[i*ind.size:(i+1)*ind.size,:] = pos[ind,:]
            atno[i*ind.size:(i+1)*ind.size] = atoms[i].get_atomic_numbers()[ind]
        #create new atoms object
        dens = Atoms(numbers=atno,positions=pos_d,cell=atoms[0].get_cell())
        return(dens)

    def _get_coordination(self,atoms,a,b,rcut):
        ''' function to obtain coordination numbers for a list of atom objects:
            input : atoms    = list of atom objects
                    a, b     = atomic numbers between which to obtain the coordination
                    rcut     = cutoff until which to obtain coordination numbers
                    min_freq = minimum share of images for an atom i to be coordinating
                               atom a to be counted into stik_set
            output: cord_dat = numpy array with dimensions (N(a),len(atoms)) giving
                               the coordination of each a in each step
                    cord_set = number of different atoms coordinating atom a across all
                               images
                    stik_dat = mean coordination duration of b around each a
            NOTE: if a = b self-coordination is included
        '''
        types = atoms[0].get_atomic_numbers()
        ind_a, ind_b = np.where(types == a)[0], np.where(types == b)[0]
        cord_dat, cord_set, stik_dat = np.zeros((ind_a.size,len(atoms))), [set_count() for a in range(0,ind_a.size)], []
        for i in range(0,len(atoms)):
            a = atoms[i]
            spos, cell = a.get_scaled_positions(), a.get_cell()
            for j in range(0,ind_a.size):
                cord = self.__get_immediate_CN(spos[ind_b,:],spos[ind_a[j],:],cell,rcut)
                cord_dat[j,i] = cord.size
                [cord_set[j].add(cord[c]) for c in range(cord.size)]
        for i in range(0,len(cord_set)):
            ids, counts = cord_set[i].get_count()
            stik_dat.append(np.mean(counts))
        cord_set = np.array([len(cord_set[c]) for c in range(len(cord_set))])
        return(cord_dat,cord_set,np.array(stik_dat))

    def _get_neighbor_inds(self,atoms,ind,rcut):
        ''' function to obtain neighbor inds within rcut
            input:  atoms     = ase-atoms-obj
                    ind       = central atom id
                    rcut      = cutoff for which to obtain points within distance
            output: inds      = neighbor ids
        '''
        pos_array = atoms.get_scaled_positions()
        cell = atoms.get_cell()
        pos = pos_array[ind,:]
        all_inds = self.__get_immediate_CN(pos_array,pos,cell,rcut)
        neighs = np.setdiff1d(all_inds,[ind],assume_unique=True)
        return(neighs)


    def __get_immediate_CN(self,pos_array,pos,cell,rcut):
        ''' function to calculate distance array (pos_array - pos) and determine
            entries within distance rcut
            input:  pos_array = positions which to calculate distances from
                    pos       = origin position
                    cell      = transformation for distance vectors
                    rcut      = cutoff for which to obtain points within distance
            output: cord      = entries of points in pos_array within distance rcut
        '''
        dvec = self._correct_vec(pos_array-pos)
        dvec = np.dot(dvec,cell)
        dist = np.linalg.norm(dvec,axis=1)
        cord = np.where(dist <= rcut)[0]
        return(cord)

    def _coordination_decay(self,atoms,a,b,rcut):
        ''' function to obtain the change/decay of the original coordination for a 
            list of atom objects/trajectory:
            input : atoms    = list of atom objects
                    a, b     = atomic numbers between which to obtain the coordination
                    rcut     = cutoff until which to obtain coordination numbers
            output: cord_ = numpy array with dimensions (N(a),len(atoms)) giving
                               the share of the original coordination in each step
            NOTE: if a = b self-coordination is included
        '''
        types = atoms[0].get_atomic_numbers()
        ind_a, ind_b = np.where(types == a)[0], np.where(types == b)[0]
        o_set, cord0  = np.zeros((ind_a.size,len(atoms))), []
        #set-up initial coordination sets:
        spos, cell = atoms[0].get_scaled_positions(), atoms[0].get_cell()
        for i in range(0,ind_a.size):
            cord0.append(self.__get_immediate_CN(spos[ind_b,:],spos[ind_a[i]],cell,rcut))
        #obtain overlap of cord0 and cord_i for each timestep i
        for i in range(0,len(atoms)):
            a = atoms[i]
            spos, cell = a.get_scaled_positions(), a.get_cell()
            for j in range(0,ind_a.size):
                cord = self.__get_immediate_CN(spos[ind_b,:],spos[ind_a[j]],cell,rcut)
                o_set[j,i] = float(np.intersect1d(cord0[j],cord).size)
        return(o_set)

    def _get_velocities_from_positions(self,atoms,timestep):
        ''' function to compute velocities from distance difference and timestep
            NOTE that this should only be done for adjacend snapshots - only for 
            orthogonal boxes
            input : atoms    = list of atom objects
                    tiemstep = timestep between snapshots
            output: vel      = list of np.arrays containing xyz velocities for N-1 snapshots
        '''
        vel = []
        for i in range(0,len(atoms)-1):
            vec = self._correct_vec(atoms[i+1].get_scaled_positions() - atoms[i].get_scaled_positions())
            vec = np.dot(vec,atoms[i].get_cell())
            vel.append(vec / timestep)
        return(vel)




