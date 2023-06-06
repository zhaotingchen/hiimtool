'''
Utility functions for analysing measurementset files.
'''

from casatools import ms,table
import numpy as np
import sys
import time
import glob
from astropy import constants,units
import datetime
import re
import pickle

def read_ms(filename,keys,sel_ch=None,verbose=False,ifraxis=False):
    '''
    A function to read in the measurementset data and returns the values.
    
    Parameters
    ----------
        filename: str 
            The name of the measurementset file to read.
        
        keys: list or array of strings 
            The keywords to read.
            
        sel_ch: float array
            The channel selection function to pass into :py:meth:`casatools.selectchannel`. 
            See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms.ms.selectchannel .
        verbose: bool, default False
            Whether to print out time information.
            
    Returns
    -------
        data: a list of np arrays containing the data corresponding to the keywords, indexed according to the input `keys`.
    '''
    if verbose:
        print('Reading data...',datetime.datetime.now().time().strftime("%H:%M:%S"))
    msset = ms()
    msset.open(filename)
    if sel_ch is not None:
        msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    data = msset.getdata(keys,ifraxis=ifraxis)
    msset.close()
    keylist = np.array(list(data.keys()))
    key_pos = np.where(np.array(keys)[:,None]==keylist[None,:])[-1]
    data = np.array(list(data.values()),dtype='object')[key_pos]
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return data