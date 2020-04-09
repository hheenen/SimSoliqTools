"""
This file contains function which can be used for job
submission

"""


def write_chain_submit(fsubmit, filehdr, execmd, cdirs, nbatch):
    """
      function to create chain-submission script
 
      Parameters
      ----------
      fsubmit : str
          name of chain submission file(s)
      filehdr : str
          header for submit file
      execmd : str
          command to execute calculation, e.g. mpirun bin.x
      cdirs : list
          list of directories to run calculations
      nbatch : int
          number of jobs to include in one chain job
    
    """
    basename = '.'.join(fsubmit.split('.')[:-1])
    suffix = fsubmit.split('.')[-1]
    
    filehdr += "\n\ncwd=$(pwd)\n\n"

    # cmd to change into sp-folder, execute calculation change back
    mdl = "\ncd %s\n{:}\ncd $cwd\n\n".format(execmd)
    
    # make nbatch - job files
    nb = list(range(0,len(cdirs),nbatch)) + [len(cdirs)]
    for i in range(len(nb) - 1):
        indirs = cdirs[nb[i]:nb[i+1]]
        ftxt = filehdr
        for indir in indirs:
            ftxt += mdl%indir
        with open('{0}_{1:02d}.{2}'.format(basename,i,suffix),'w') as sf:
            sf.write(ftxt)


