from casatools import ms,table
import numpy as np
import sys
import time
from astropy.cosmology import Planck18_arXiv_v2 as Planck18
import glob
from astropy import constants,units
import datetime

#fill=True
#verbose=True


def slicer_vectorized(a,start,end):
    """A function for slicing through numpy arrays with string elements"""
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.frombuffer(b.tobytes(),dtype=(str,end-start))

f_21 = 1420405751.7667 # in Hz
#num_ch=220
#flag_frac = 0.2 # if flagged fraction larger than this, throw away
#ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
#uvedges = np.linspace(-6000,6000,201)-30
save_dir = "/idia/projects/mightee/zchen/vis_grid/deep2/"
ms_dir = '/idia/projects/mightee/DEEP2_data/wselfcal/'

def worker(scan_indx,submslist,uvedges,flag_frac,sel_ch,fill=False,verbose=False):
    '''
    Returns the gridded visibility sum and the baseline number counts of a given scan for a ms file.
    Note that the output is **SUMMED** visibility not average!

    Parameters
    ----------
        scan_indx: int 
            The indx of the subms file (scan) to read.
        
        submsliat: string array 
            The list of the subms files corresponding to each scan
        
        uvedges: numpy array
            The edges of the uv grids in wavelength unit (dimensionless, not in metres)
        
        flag_frac: float
            If a baseline has frag_frac or larger fraction of frequency channels flagged, then this baseline is neglected
        
        sel_ch: float array
            The channel selection function to pass into :py:meth:`casatools.selectchannel`. 
            See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms.ms.selectchannel .
        
        fill: Boolean, default False
            Whether to fill flagged channels (after `frag_frac` exclusion) with the nearest neighbours
        
        verbose: Boolean, default False
            Whether to print information about time and which block and scan the function is reading.

    Returns
    -------
        vis_gridded: complex array of shape (num_ch,num_uv,num_uv).
        count: float array. If `fill=True` then shape is (num_uv,num_uv), else (num_ch,num_uv,num_uv).
    '''
    # note that here the output is the sum of the vis not the average
    num_ch = sel_ch[0]
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    filename = submslist[scan_indx]
    msset = ms()
    msset.open(filename)
    msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]] # get the frequencies
    assert len(freq_arr) == num_ch
    #get all the data needed
    if verbose:
        print('Reading data...',datetime.datetime.now().time().strftime("%H:%M:%S"))
    data = msset.getdata('corrected_data')['corrected_data']
    uarr = msset.getdata('u')['u']
    varr = msset.getdata('v')['v']
    flag = msset.getdata('flag')['flag']
    msset.close()
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    #data = data[:,:num_ch,:]
    #flag = flag[:,:num_ch,:]
    #z_arr = f_21/freq_arr-1 # redshifts
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    #z_0 = (z_arr[0]+z_arr[-1])/2 # effective redshifts
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    
    if verbose:
        print('Gridding...',datetime.datetime.now().time().strftime("%H:%M:%S"))
    # meter to wavelength
    uarr /= lamb_0
    varr /= lamb_0
    flag_I = (flag[0]+flag[-1])>0  # if XX or YY is flagged then I is flagged
    indx = flag_I.mean(axis=0)<flag_frac
    #get rid of the flag_frac excluded channels
    data = data[:,:,indx] 
    flag = flag[:,:,indx]
    uarr = uarr[indx]
    varr = varr[indx]
    flag_I = flag_I[:,indx]
    
    if fill:
        for p_indx in (0,3): # XX and YY
            farr,rowarr = np.where(flag[p_indx]==1) # find the flags
            # black magic for finding the nearest unflagged channel
            # this line is designed to give divided by zero. suppress that specific warning.
            with np.errstate(divide='ignore'):
                freqfill = np.argmin(
                    np.nan_to_num(
                        np.abs(farr[None,:]-ch_arr[:,None])*(1-flag[p_indx][:,rowarr])/(1-flag[p_indx][:,rowarr])
                        ,nan=np.inf
                    )
                    ,axis=0
                )
            data[p_indx,farr,rowarr] = data[p_indx,freqfill,rowarr] 
        flag_I = np.zeros_like(flag_I) # if filled then all the channels are used
    
    data = (data[0]+data[-1])/2 # just keep I
    vis_gridded = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    # if filled, then all the channels have same number of baselines in each u-v grid
    if fill:
        count = np.zeros((len(uvedges)-1,len(uvedges)-1)) 
    else:
        count = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
    
    # grid the vis
    for i in range(num_ch):
        vis_real,_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],
                                      weights=(data[i].real)*(1-flag_I[i]))
        vis_imag,_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],
                                      weights=(data[i].imag)*(1-flag_I[i]))
        vis_gridded[i] += vis_real+1j*vis_imag
        if fill:
            continue
        else:
            count[i],_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],weights=(1-flag_I[i]))
    if fill:
        count,_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],)
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return vis_gridded,count

def save_scan(i,args):
    '''
    A wrapper function for prompting :function:`worker` and save the gridded visibility summations and counts into a scratch folder.
    The files are saved in the path `scratch_dir` with the format '(vis/count)_blockid_scanid.npy'.
    Note that if you are running multiple gridding function on the same timeblock you may need to be careful.
    Save them in different scratch folder to avoid overriding.

    Parameters
    ----------
        i: int 
            The indx of the subms file (scan) to read.
        
        args: list
            See the below input list for each element from first to last
        
        submsliat: string array 
            The list of the subms files corresponding to each scan
        
        uvedges: numpy array
            The edges of the uv grids in wavelength unit (dimensionless, not in metres)
        
        frac: float
            If a baseline has frag_frac or larger fraction of frequency channels flagged, then this baseline is neglected
            
        block_num: str
            The 10-digit data block id. Purely for file name formatting, the actual data block should be specified in `submslist`.
        
        scratch_dir: str
            The path to the scratch folder to store the output.
        
        sel_ch: float array
            The channel selection function to pass into :py:meth:`casatools.selectchannel`. 
            See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms.ms.selectchannel .
        
        fill: Boolean, default False
            Whether to fill flagged channels (after `frag_frac` exclusion) with the nearest neighbours
        
        verbose: Boolean, default False
            Whether to print information about time and which block and scan the function is reading.
        
        

    Returns
    -------
        1
    '''
    submslist,uvedges,frac,block_num,scratch_dir,sel_ch,fill,verbose = args
    scan_id = slicer_vectorized(submslist,-7,-3)
    if verbose:
        print('Reading block', block_num,'Scan', scan_id[i],datetime.datetime.now().time().strftime("%H:%M:%S"))
    vis_i, count_i = worker(i,submslist,uvedges,frac,sel_ch,fill=fill,verbose=verbose)
    np.save(scratch_dir+'vis_'+block_num+'_'+scan_id[i],vis_i)
    np.save(scratch_dir+'count_'+block_num+'_'+scan_id[i],count_i)
    if verbose:
        print('Block', block_num,'Scan', scan_id[i],'finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return 1
