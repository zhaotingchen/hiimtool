from casatools import ms,table
import numpy as np
import sys
import time
import glob
from astropy import constants,units
import datetime
import re

#fill=True
#verbose=True


def slicer_vectorized(a,start,end):
    """A function for slicing through numpy arrays with string elements"""
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.frombuffer(b.tobytes(),dtype=(str,end-start))

f_21 = 1420405751.7667 # in Hz

def read_ms(filename,keys,sel_ch,verbose=False):
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
    msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    data = msset.getdata(keys)
    msset.close()
    keylist = np.array(list(data.keys()))
    key_pos = np.where(np.array(keys)[:,None]==keylist[None,:])[-1]
    data = np.array(list(data.values()))[key_pos]
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return data

def fill_row(flag):
    '''
    black magic for finding the nearest unflagged channel.
    
    Parameters
    ----------
        flag: bool array. 
            The data flags. must of in the shape of (num_channels,num_baselines)
            
    Returns
    -------
        freqfill: int array.
            The nearest unflagged frequency channel for each flagged data point.
        farr: int array.
            The frequency channel of flagged data point.
        rowarr: int array.
            The baseline index of flagged data point.
    '''
    # black magic for finding the nearest unflagged channel
    num_ch = len(flag)
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    farr,rowarr = np.where(flag==1) # find the flags
    # this line is designed to give divided by zero. suppress that specific warning.
    with np.errstate(invalid='print'):
        freqfill = np.argmin(
            np.nan_to_num(
                np.abs(farr[None,:]-ch_arr[:,None])*(1-flag[:,rowarr])/(1-flag[:,rowarr])
                ,nan=np.inf
            ),axis=0
        )
    return freqfill,farr,rowarr


def sumamp(scan_indx,submslist,uvedges,flag_frac,sel_ch,stokes='I',fill=False,verbose=False):
    '''
    Returns the SUM of the amplitude square of the visibilities, useful for checking noise rms.
    
    Parameters
    ----------
        scan_indx: int 
            The indx of the subms file (scan) to read.
        
        submsliat: string array 
            The list of the subms files corresponding to each scan
        
        uvedges: numpy array
            The edges of the uv grids in wavelength unit (dimensionless, not in metres). The array should be formatted as [umin,umax,vmin,vmax] or [|u|min,|u|max]
        
        flag_frac: float
            If a baseline has frag_frac or larger fraction of frequency channels flagged, then this baseline is neglected
        
        sel_ch: float array
            The channel selection function to pass into :py:meth:`casatools.selectchannel`. 
            See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms.ms.selectchannel .
        
        stokes: string, default "I"
            Which Stokes parameter to grid. Currently only support I and V.
            
        fill: Boolean, default False
            Whether to fill flagged channels (after `frag_frac` exclusion) with the nearest neighbours
        
        verbose: Boolean, default False
            Whether to print information about time and which block and scan the function is reading.

    Returns
    -------
        vis_sum: float.
        count: float.
    '''
    if stokes=='I':
        pol = (0,3)
        sign = np.array([1,1])
    else:
        pol = (1,2)
        sign = np.array([-1j,1j])
        
    num_ch = sel_ch[0]
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    filename = submslist[scan_indx]
    msset = ms()
    msset.open(filename)
    msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]] # get the frequencies
    msset.close()
    assert len(freq_arr) == num_ch
    #get all the data needed
    data,uarr,varr,flag = read_ms(submslist[scan_indx],['corrected_data','u','v','flag'],sel_ch,verbose)
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    
    if verbose:
        print('Calculating...',datetime.datetime.now().time().strftime("%H:%M:%S"))
    # meter to wavelength
    uarr /= lamb_0
    varr /= lamb_0
    flag_I = (flag[pol[0]]+flag[pol[1]])>0  # if XX or YY is flagged then I is flagged 
    indx = flag_I.mean(axis=0)<flag_frac
    #get rid of the flag_frac excluded channels
    data = data[:,:,indx] 
    flag = flag[:,:,indx]
    uarr = uarr[indx]
    varr = varr[indx]
    flag_I = flag_I[:,indx]
    if len(uvedges) == 4:
        indx = (uarr>uvedges[0])*(uarr<uvedges[1])*(varr>uvedges[2])*(varr<uvedges[3])
    else:
        umodearr = np.sqrt(uarr**2+varr**2)
        indx = (umodearr>uvedges[0])*(umodearr<uvedges[1])
    data = data[:,:,indx]
    flag_I = flag_I[:,indx]
    data = (data[pol[0]]*sign[0]+data[pol[1]]*sign[1])/2 # just keep I
    vis_sum = (np.abs(data*(1-flag_I))**2).sum()
    count = (1-flag_I).sum()
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return vis_sum,count
    

def worker(scan_indx,submslist,uvedges,flag_frac,sel_ch,stokes='I',col='corrected_data',fill=False,verbose=False):
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
        
        stokes: string, default "I"
            Which Stokes parameter to grid. Currently only support I and V.
        
        col: string, default "corrected_data"
            Which data column to read. Default reads the corrected_data column after selfcal.
            
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
    if stokes=='I':
        pol = (0,3)
        sign = np.array([1,1])
    else:
        pol = (1,2)
        sign = np.array([-1j,1j])
        
    num_ch = sel_ch[0]
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    filename = submslist[scan_indx]
    msset = ms()
    msset.open(filename)
    msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]] # get the frequencies
    assert len(freq_arr) == num_ch
    msset.close()
    #get all the data needed
    data,uarr,varr,flag = read_ms(submslist[scan_indx],[col,'u','v','flag'],sel_ch,verbose)
    
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    
    if verbose:
        print('Gridding...',datetime.datetime.now().time().strftime("%H:%M:%S"))
    # meter to wavelength
    uarr /= lamb_0
    varr /= lamb_0
    flag_I = (flag[pol[0]]+flag[pol[1]])>0  # if XX or YY is flagged then I is flagged 
    indx = flag_I.mean(axis=0)<flag_frac
    #get rid of the flag_frac excluded channels
    data = data[:,:,indx] 
    flag = flag[:,:,indx]
    uarr = uarr[indx]
    varr = varr[indx]
    flag_I = flag_I[:,indx]
    
    if fill:
        for p_indx in pol: # XX and YY
            freqfill,farr,rowarr = fill_row(flag[p_indx])
            #farr,rowarr = np.where(flag[p_indx]==1) # find the flags
            # black magic for finding the nearest unflagged channel
            # this line is designed to give divided by zero. suppress that specific warning.
            #with np.errstate(divide='ignore'):
            #    freqfill = np.argmin(
            #        np.nan_to_num(
            #            np.abs(farr[None,:]-ch_arr[:,None])*(1-flag[p_indx][:,rowarr])/(1-flag[p_indx][:,rowarr])
            #            ,nan=np.inf
            #        )
            #        ,axis=0
            #    )
            data[p_indx,farr,rowarr] = data[p_indx,freqfill,rowarr] 
        flag_I = np.zeros_like(flag_I) # if filled then all the channels are used
    
    data = (data[pol[0]]*sign[0]+data[pol[1]]*sign[1])/2 # just keep I
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
        
        submslist: string array 
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
        
        stokes: string, default "I"
            Which Stokes parameter to grid. Currently only support I and V.
            
        col: string, default "corrected_data"
            Which data column to read. Default reads the corrected_data column after selfcal.
            
        fill: Boolean, default False
            Whether to fill flagged channels (after `frag_frac` exclusion) with the nearest neighbours
        
        verbose: Boolean, default False
            Whether to print information about time and which block and scan the function is reading.
        
        

    Returns
    -------
        1
    '''
    submslist,uvedges,frac,block_num,scratch_dir,sel_ch,stokes,col,fill,verbose = args
    scan_id = slicer_vectorized(submslist,-7,-3)
    if verbose:
        print('Reading block', block_num,'Scan', scan_id[i],datetime.datetime.now().time().strftime("%H:%M:%S"))
    vis_i, count_i = worker(i,submslist,uvedges,frac,sel_ch,stokes,col=col,fill=fill,verbose=verbose)
    np.save(scratch_dir+'vis_'+block_num+'_'+scan_id[i]+'_'+stokes,vis_i)
    np.save(scratch_dir+'count_'+block_num+'_'+scan_id[i]+'_'+stokes,count_i)
    if verbose:
        print('Block', block_num,'Scan', scan_id[i],'finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return 1



def sum_cal(scan_indx,cal_tab,submslist,uvedges,flag_frac,sel_ch,stokes='I',fill=False,verbose=False):
    '''
    Returns the SUM of the amplitude of the calibration errors.
    
    Parameters
    ----------
        scan_indx: int 
            The indx of the subms file (scan) to read.
        
        cal_tab: numpy array
            The calibration solution of the observation block.
            cal_tab[0] should be the times of each solution interval in increasing order.
            cal_tab[1] is the time-ordered id (from 0 to n-1 for n solution intervals) for each sol.
            cal_tab[2] is the randomised error based on the ``valueErr`` column for each sol at each frequency.
            cal_tab[3] is the spline fitting error based on the deviation of the solution to the spline for each sol at each frequency.
            cal_tab[4] is the feed id ("1" or "2") for each sol
            cal_tab[5] is the antenna id for each sol.
        
        submsliat: string array 
            The list of the subms files corresponding to each scan
        
        uvedges: numpy array
            The edges of the uv grids in wavelength unit (dimensionless, not in metres). The array should be formatted as [umin,umax,vmin,vmax] or [|u|min,|u|max]
        
        flag_frac: float
            If a baseline has frag_frac or larger fraction of frequency channels flagged, then this baseline is neglected
        
        sel_ch: float array
            The channel selection function to pass into :py:meth:`casatools.selectchannel`. 
            See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms.ms.selectchannel .
        
        stokes: string, default "I"
            Which Stokes parameter to grid. Currently only support I and V.
            
        fill: Boolean, default False
            Whether to fill flagged channels (after `frag_frac` exclusion) with the nearest neighbours
        
        verbose: Boolean, default False
            Whether to print information about time and which block and scan the function is reading.

    Returns
    -------
        vis_sum: float.
        count: float.
    '''
    if stokes=='I':
        pol = (0,3)
        sign = np.array([1,1])
    else:
        pol = (1,2)
        sign = np.array([-1j,1j])
    cal_tstep,cal_tid,cal_gerr,cal_ferr,cal_feed,cal_ant = cal_tab    
    num_ch = sel_ch[0]
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    filename = submslist[scan_indx]
    msset = ms()
    msset.open(filename)
    msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]] # get the frequencies
    assert len(freq_arr) == num_ch
    msset.close()
    ant1,ant2,timearr,uarr,varr,flag = read_ms(filename,['antenna1','antenna2','time','u','v','flag'],sel_ch,verbose)

    data_valerr = np.zeros((4,num_ch,len(uarr))) # based on valErr
    data_splerr = np.zeros((4,num_ch,len(uarr))) # based on spline fitting
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    #z_0 = (z_arr[0]+z_arr[-1])/2 # effective redshifts
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    if verbose:
        print('Calculating...',datetime.datetime.now().time().strftime("%H:%M:%S"))

    data_tid = (timearr[:,None]>cal_tstep[None,:]).sum(axis=-1)-1
    sel_ant1X = (cal_tid[:,None] == data_tid[None,:])*(cal_ant[:,None] == ant1[None,:])*(cal_feed == '1')[:,None]
    sel_ant2X = (cal_tid[:,None] == data_tid[None,:])*(cal_ant[:,None] == ant2[None,:])*(cal_feed == '1')[:,None]
    sel_ant1Y = (cal_tid[:,None] == data_tid[None,:])*(cal_ant[:,None] == ant1[None,:])*(cal_feed == '2')[:,None]
    sel_ant2Y = (cal_tid[:,None] == data_tid[None,:])*(cal_ant[:,None] == ant2[None,:])*(cal_feed == '2')[:,None]
    
    #check only one point gets selected
    assert np.product(sel_ant1X.sum(axis=0)==1) == 1
    assert np.product(sel_ant1Y.sum(axis=0)==1) == 1
    assert np.product(sel_ant2X.sum(axis=0)==1) == 1
    assert np.product(sel_ant2Y.sum(axis=0)==1) == 1
    
    sol_indx1X = np.where(sel_ant1X)[0][np.argsort(np.where(sel_ant1X)[1])]
    sol_indx1Y = np.where(sel_ant1Y)[0][np.argsort(np.where(sel_ant1Y)[1])]
    sol_indx2X = np.where(sel_ant2X)[0][np.argsort(np.where(sel_ant2X)[1])]
    sol_indx2Y = np.where(sel_ant2Y)[0][np.argsort(np.where(sel_ant2Y)[1])]

    data_valerr = np.array(
        [
        cal_gerr[sol_indx1X]*cal_gerr[sol_indx2X],
        cal_gerr[sol_indx1X]*cal_gerr[sol_indx2Y],
        cal_gerr[sol_indx1Y]*cal_gerr[sol_indx2X],
        cal_gerr[sol_indx1Y]*cal_gerr[sol_indx2Y]
        ]
    )
    data_valerr = np.transpose(data_valerr,axes=(0,-1,-2))
    data_splerr = np.array(
        [
        cal_ferr[sol_indx1X]*cal_ferr[sol_indx2X],
        cal_ferr[sol_indx1X]*cal_ferr[sol_indx2Y],
        cal_ferr[sol_indx1Y]*cal_ferr[sol_indx2X],
        cal_ferr[sol_indx1Y]*cal_ferr[sol_indx2Y]
        ]
    )
    data_splerr = np.transpose(data_splerr,axes=(0,-1,-2))
    # meter to wavelength
    uarr /= lamb_0
    varr /= lamb_0
    flag_I = (flag[pol[0]]+flag[pol[1]])>0  # if XX or YY is flagged then I is flagged 
    indx = flag_I.mean(axis=0)<flag_frac
    #get rid of the flag_frac excluded channels
    data_valerr = data_valerr[:,:,indx] 
    data_splerr = data_splerr[:,:,indx]
    uarr = uarr[indx]
    varr = varr[indx]
    flag_I = flag_I[:,indx]
    flag = flag[:,:,indx]
    
    if fill:
        for p_indx in pol: # XX and YY
            freqfill,farr,rowarr = fill_row(flag[p_indx])
            data_valerr[p_indx,farr,rowarr] = data_valerr[p_indx,freqfill,rowarr] 
            data_splerr[p_indx,farr,rowarr] = data_splerr[p_indx,freqfill,rowarr]
        flag_I = np.zeros_like(flag_I) # if filled then all the channels are used
    
    data_valerr = (data_valerr[pol[0]]*sign[0]+data_valerr[pol[1]]*sign[1])/2 # just keep I
    data_splerr = (data_splerr[pol[0]]*sign[0]+data_splerr[pol[1]]*sign[1])/2 # just keep I
    valerr_gridded = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
    splerr_gridded = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
    # if filled, then all the channels have same number of baselines in each u-v grid
    if fill:
        count = np.zeros((len(uvedges)-1,len(uvedges)-1)) 
    else:
        count = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
    
    # grid the vis
    for i in range(num_ch):
        valerr_gridded[i],_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],
                                      weights=(data_valerr[i])*(1-flag_I[i]))
        splerr_gridded[i],_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],
                                      weights=(data_splerr[i])*(1-flag_I[i]))
        if fill:
            continue
        else:
            count[i],_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],weights=(1-flag_I[i]))
    if fill:
        count,_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],)
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return valerr_gridded,splerr_gridded,count


def save_cal(i,args):
    '''
    A wrapper function for prompting :function:`sum_cal` and save the gridded calibration error summations and counts into a scratch folder.
    The files are saved in the path `scratch_dir` with the format '(vis/count)_blockid_scanid.npy'.
    Note that if you are running multiple gridding function on the same timeblock you may need to be careful.
    Save them in different scratch folder to avoid overriding.

    Parameters
    ----------
        i: int 
            The indx of the subms file (scan) to read.
        
        args: list
            See the below input list for each element from first to last
        
        cal_tab: numpy array
            The calibration solution of the observation block.
            cal_tab[0] should be the times of each solution interval in increasing order.
            cal_tab[1] is the time-ordered id (from 0 to n-1 for n solution intervals) for each sol.
            cal_tab[2] is the randomised error based on the ``valueErr`` column for each sol at each frequency.
            cal_tab[3] is the spline fitting error based on the deviation of the solution to the spline for each sol at each frequency.
            cal_tab[4] is the feed id ("1" or "2") for each sol
            cal_tab[5] is the antenna id for each sol.
        
        submsliat: string array 
            The list of the subms files corresponding to each scan
        
        uvedges: numpy array
            The edges of the uv grids in wavelength unit (dimensionless, not in metres). The array should be formatted as [umin,umax,vmin,vmax] or [|u|min,|u|max]
        
        frac: float
            If a baseline has frag_frac or larger fraction of frequency channels flagged, then this baseline is neglected
        
        block_num: str
            The 10-digit data block id. Purely for file name formatting, the actual data block should be specified in `submslist`.
        
        scratch_dir: str
            The path to the scratch folder to store the output.
        
        sel_ch: float array
            The channel selection function to pass into :py:meth:`casatools.selectchannel`. 
            See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms.ms.selectchannel .
        
        stokes: string, default "I"
            Which Stokes parameter to grid. Currently only support I and V.
            
        fill: Boolean, default False
            Whether to fill flagged channels (after `frag_frac` exclusion) with the nearest neighbours
        
        verbose: Boolean, default False
            Whether to print information about time and which block and scan the function is reading.
        

    Returns
    -------
        1
    '''
    cal_tab,submslist,uvedges,frac,block_num,scratch_dir,sel_ch,stokes,fill,verbose = args
    scan_id = slicer_vectorized(submslist,-7,-3)
    if verbose:
        print('Reading block', block_num,'Scan', scan_id[i],datetime.datetime.now().time().strftime("%H:%M:%S"))
    varerr_i,splerr_i,count_i = sum_cal(i,cal_tab,submslist,uvedges,frac,sel_ch,stokes=stokes,fill=fill,verbose=verbose)
    np.save(scratch_dir+'varerr_'+block_num+'_'+scan_id[i]+'_'+stokes,varerr_i)
    np.save(scratch_dir+'splerr_'+block_num+'_'+scan_id[i]+'_'+stokes,splerr_i)
    np.save(scratch_dir+'count_'+block_num+'_'+scan_id[i]+'_'+stokes,count_i)
    if verbose:
        print('Block', block_num,'Scan', scan_id[i],'finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return 1

def find_block_id(filename):
    reex = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
    result = re.findall(reex,filename)
    # make sure there is only one block_id in the path
    assert len(result)==1
    result = result[0]
    return result

vfind_id = np.vectorize(find_block_id)