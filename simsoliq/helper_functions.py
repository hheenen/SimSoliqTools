"""
This module contains general helper functions

"""

import pickle, os
import hashlib, base64

def load_pickle_file(filename,py23=False):
    pickle_file = open(filename,'rb')
    if py23:
        data = pickle.load(pickle_file, encoding='latin1')
    else:
        data = pickle.load(pickle_file)
    pickle_file.close()
    return(data)

def write_pickle_file(filename,data,py2=False):
    ptcl = 0
    if py2:
        filename = filename.split('.')[0]+'_py2.'+filename.split('.')[1]
        ptcl = 2
    output = open(filename, 'wb')
    pickle.dump(data, output, protocol=ptcl)
    output.close()

def pkl_decorator_factory(pklname):
    """
      decorator to save function output in pklfile in `self.bpath`

      Parameters
      ----------
      pklname : str
        name of pklfile being saved at: self.bpath/`pklname`.pkl
    
    """
    def pkl_decorator(func):
        def pkl_wrapper(*args, **kwargs):
            pklfile = args[0].bpath+'/%s.pkl'%pklname
            # determine savepkl argument
            savepkl = True
            if 'savepkl' in kwargs:
                savepkl = kwargs['savepkl']
            # process data
            if not savepkl or not os.path.isfile(pklfile):
                output = func(*args,**kwargs)
            # read pklfile
            elif savepkl:
                output = load_pickle_file(pklfile)
            # save pklfile
            if savepkl:
                write_pickle_file(pklfile, output)
            return(output)
        return(pkl_wrapper)
    return(pkl_decorator)
    
def _arr2hash(arr):
    arr[arr == 0] = 0
    hashed = hashlib.md5(arr).digest()
    hashed = base64.urlsafe_b64encode(hashed)
    return(hashed)


