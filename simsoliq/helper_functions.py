"""
This module contains general helper functions

"""

import pickle

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

