"""
This module contains functionality for creating an mdtraj-object
capable of reading VASP formats

"""

from simsoliq.io.data_mdtraj import DataMDtraj

class MDtraj(DataMDtraj):
    """ MDtraj object
      
      central object of data-manipulation in simsoliq
      derived from DataMDtraj which manages data-handling
      
      Parameters
      ----------
      See simsoliq.io.data_mdtraj.DataMDtraj for parser
      options to pass
      
      The MDtraj object is most conveniently created via the `init_mdtraj`
      function (simsolig.io.init_mdtraj). The MDtraj object can be used
      for data manipulation and analysis.
      
      Example
      -------
      # creation of mdtraj-object
      from simsoliq.io import init_mdtraj
      mdtraj = init_mdtraj("path_to_mddata", fmat='vasp')

      # retrieving energies
      epot = mdtraj.get_potential_energies()
      ...
    
    """
    def __init__(self, bpath, fident, concmode=None, fnest=None, \
                 safe_asetraj_files=True):
        # set-up DataMDtraj
        DataMDtraj.__init__(self, bpath, fident, \
            concmode=concmode, fnest=fnest, \
            safe_asetraj_files=safe_asetraj_files)
        # other options to follow ...


