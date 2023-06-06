from casatools import ms,table
import numpy as np
import sys
import time
import glob
from astropy import constants,units
import datetime
import re
import pickle
import warnings
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin
from scipy.signal import blackmanharris

def chisq(pars,func,xarr,yarr,yerr):
    #fitfunc,xarr,yarr,yerr = args
    fitarr = func(xarr,pars)
    return 0.5*np.sum((fitarr-yarr)**2/yerr**2)

def fitfunc(xarr,pars):
    '''
    (a|x|+1)*exp(-|x|/b)+c
    '''
    a,b,c = pars
    xarr = np.abs(xarr)
    return np.exp(-(xarr/b)**a)*(1-c)+c

def sort_ifr(time,ant1,ant2):
    '''
    Manually sort the ms data into shapes of (num_time,num_ifr)
    '''
    time_step = np.unique(time)
    ant_id = np.unique(np.append(ant1,ant2))
    ifr_num = ant1*1000+ant2
    dt_min = np.diff(time_step[np.argsort(time_step)]).min()
    sort_arr = time+ dt_min*ifr_num/(ifr_num.max()+1)
    sort_indx = np.argsort(sort_arr)
    return sort_indx,len(time_step),int(len(ant_id)*(len(ant_id)-1)/2)


class Specs:
    """
    Defines the basic specifications of simulation.
    """
    def __init__(self, 
                 cosmo,
                 freq_start_hz,
                 num_channels,
                 deltav_ch,
                 FWHM_ref,
                 FWHM_freq_ref,
                ):
        self.cosmo = cosmo
        self.freq_start_hz = freq_start_hz # in Hz
        self.num_channels = num_channels
        self.deltav_ch = deltav_ch # in Hz
        self.FWHM_ref = FWHM_ref
        self.FWHM_freq_ref = FWHM_freq_ref

        
    def freq_arr(self):
        """The observing frequency of each channel"""
        arr = (self.freq_start_hz + 
               np.arange(self.num_channels)*self.deltav_ch)
        return arr
    
    def FWHM_arr(self):
        """The FWHM of each channel"""
        return self.FWHM_ref*self.FWHM_freq_ref/self.freq_arr()
    
    def sigma_arr(self):
        """The beam radius of each channel"""
        return self.FWHM_arr()/2/np.sqrt(2*np.log(2))
    
    def lamb_arr(self):
        """The observing wavelength of each channel"""
        return constants.c.value/self.freq_arr()
    
    def z_arr(self):
        """The observing redshift of each channel"""
        return f_21 / self.freq_arr() -1
    
    def chi_arr(self):
        """the comoving distance of each channel"""
        return self.cosmo.comoving_distance(self.z_arr()).value
    
    def Hz_arr(self):
        """The Hubble parameter of each channel"""
        return self.cosmo.H(self.z_arr()).value
    
    def eta_arr(self):
        """Array of eta from Fourier transform in Hz^-1"""
        return np.fft.fftfreq(self.num_channels,d=self.deltav_ch)
    
    def z_0(self):
        """The mean redshift of the frequency range"""
        z_arr = self.z_arr()
        return (z_arr[0]+z_arr[-1])/2
    
    def freq_0(self):
        """The frequency of the mean redshift"""
        return f_21/(1+self.z_0())
    
    def sigma_0(self):
        """The beam size corresponding to the mean redshift"""
        sigma_0 = (self.FWHM_ref*self.FWHM_freq_ref
                   /self.freq_0()/2/np.sqrt(2*np.log(2)))
        return sigma_0
    
    def X_0(self):
        """The comoving radial distance at z_0"""
        return self.cosmo.comoving_distance(self.z_0()).value
    
    def Y_0(self):
        """The los distance per frequency, Mpc/Hz"""
        z_0 = self.z_0()
        Y_0 = ((constants.c/self.cosmo.H(z_0)/
                (f_21*units.Hz)*(1+z_0)**2)).to('Mpc/Hz').value # in Mpc/Hz
        return Y_0
    
    def lambda_0(self):
        """The wavelength at z_0"""
        return constants.c.value/self.freq_0()
    
    def k_para(self):
        """The line-of-sight k_para given by 2*pi*eta/Y"""
        return 2*np.pi*self.eta_arr()/self.Y_0()
    
    def k_perp(self,umode):
        """Calculate the transverse scale corresponding to the u-v radius"""
        return 2*np.pi*umode/self.X_0()


def slicer_vectorized(a,start,end):
    """A function for slicing through numpy arrays with string elements"""
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.frombuffer(b.tobytes(),dtype=(str,end-start))

f_21 = 1420405751.7667 # in Hz

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
    with np.errstate(divide='ignore', invalid='ignore'):
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
    

def worker(scan_indx,submslist,uvedges,flag_frac,sel_ch,stokes='I',col='corrected_data',fill=False,verbose=False,ignore_flag=False):
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
        
        ignore_flag: Boolean, default False
            Whether to ignore the flags, useful for not applying flags to the model visibility.

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
        
    filename = submslist[scan_indx]
    msset = ms()
    msset.open(filename)
    if sel_ch is not None:
        msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    if sel_ch is not None:
        freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]] # get the frequencies
    else:
        freq_arr = metadata.chanfreqs(0)
    msset.close()
    num_ch = len(freq_arr)    

    #get all the data needed
    data,uarr,varr,flag = read_ms(submslist[scan_indx],[col,'u','v','flag'],sel_ch,verbose)
    
    if ignore_flag is True:
        flag = np.zeros_like(flag)
    
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
        
        ignore_flag: Boolean, default False
            Whether to ignore the flags, useful for not applying flags to the model visibility.
        

    Returns
    -------
        1
    '''
    submslist,uvedges,frac,block_num,scratch_dir,sel_ch,stokes,col,fill,verbose,ignore_flag = args
    scan_id = vfindscan(submslist)
    block_id = vfind_id(submslist)
    if verbose:
        print('Reading block', block_id[i],'Scan', scan_id[i],datetime.datetime.now().time().strftime("%H:%M:%S"))
    vis_i, count_i = worker(i,submslist,uvedges,frac,sel_ch,stokes,col=col,fill=fill,verbose=verbose,ignore_flag=ignore_flag)
    np.save(scratch_dir+'vis_'+block_id[i]+'_'+scan_id[i]+'_'+stokes,vis_i)
    np.save(scratch_dir+'count_'+block_id[i]+'_'+scan_id[i]+'_'+stokes,count_i)
    if verbose:
        print('Block', block_id[i],'Scan', scan_id[i],'finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
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
    scan_id = vfindscan(submslist)
    block_id = vfind_id(submslist)
    if verbose:
        print('Reading block', block_id[i],'Scan', scan_id[i],datetime.datetime.now().time().strftime("%H:%M:%S"))
    varerr_i,splerr_i,count_i = sum_cal(i,cal_tab,submslist,uvedges,frac,sel_ch,stokes=stokes,fill=fill,verbose=verbose)
    np.save(scratch_dir+'varerr_'+block_id[i]+'_'+scan_id[i]+'_'+stokes,varerr_i)
    np.save(scratch_dir+'splerr_'+block_id[i]+'_'+scan_id[i]+'_'+stokes,splerr_i)
    np.save(scratch_dir+'count_'+block_id[i]+'_'+scan_id[i]+'_'+stokes,count_i)
    if verbose:
        print('Block', block_id[i],'Scan', scan_id[i],'finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return 1

def get_rfisum(scan_indx,block_indx,block_id,submslist,save_dir,col,sel_ch,window,flag_frac,hor_low,hor_high,verbose):
    """
    A function to return a collection of diagnostic statistics for identifying the residual RFI near the horizon.
    To be expanded.
    """
    scan_id = vfindscan(submslist)
    if verbose:
        print('Now processing Block '+block_id[block_indx]+' Scan '+scan_id[scan_indx],datetime.datetime.now().time().strftime("%H:%M:%S"))
    msset = ms()
    msset.open(submslist[scan_indx])
    msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]] # get the frequencies
    num_ch = sel_ch[0]
    msset.close()
    delta_ch = metadata.chanres(spw=0)[0]
    data,uarr,varr,flag,timearr,ifr_arr = read_ms(submslist[scan_indx],[col,'u','v','flag','time','ifr_number'],sel_ch,verbose)
    time_step = np.unique(timearr)
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    sign_I = np.array([1,1])
    sign_V = np.array([-1j,1j])
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    uarr /= lamb_0
    varr /= lamb_0
    flag_I = (flag[0]+flag[-1])>0  # if XX or YY is flagged then I is flagged 
    indx_I = flag_I.mean(axis=0)<flag_frac
    flag_V = (flag[1]+flag[2])>0  # if XX or YY is flagged then I is flagged 
    indx_V = flag_V.mean(axis=0)<flag_frac
    flag_I = 0
    flag_V = 0
    data_I = np.array([data[0][:,indx_I],data[-1][:,indx_I]])
    data_V = np.array([data[1][:,indx_V],data[2][:,indx_V]])
    if verbose:
        print("Inpainting...",datetime.datetime.now().time().strftime("%H:%M:%S"))
    for i,p_indx in enumerate((0,3)): # XX and YY
        freqfill,farr,rowarr = fill_row(flag[p_indx][:,indx_I]) # find the flags
        data_I[i][farr,rowarr] = data[p_indx][:,indx_I][freqfill,rowarr]
    for i,p_indx in enumerate((1,2)): # XY and YX
        freqfill,farr,rowarr = fill_row(flag[p_indx][:,indx_V]) # find the flags
        data_V[i][farr,rowarr] = data[p_indx][:,indx_V][freqfill,rowarr]
    if verbose:
        print("...Finished",datetime.datetime.now().time().strftime("%H:%M:%S"))
    data_I = (data_I[0]*sign_I[0]+data_I[1]*sign_I[1])/2
    data_V = (data_V[0]*sign_V[0]+data_V[1]*sign_V[1])/2
    data = 0
    V_avg = np.zeros((len(time_step),len(freq_arr)))
    for time_i in range(len(time_step)):
        V_avg[time_i] = np.sqrt(np.mean(np.abs(data_V[:,timearr[indx_V]==time_step[time_i]])**2,axis=-1))
    sigma_ch = np.sqrt((np.abs(data_V)**2).mean(axis=-1))
    if np.isnan(sigma_ch).sum()>0:
        sigma_ch = fill_nan(sigma_ch)
    
    eta_arr = np.fft.fftfreq(220,d=delta_ch) # in seconds
    eta_arr = np.fft.fftshift(eta_arr)
    # delay transform
    testarr_f = np.zeros(num_ch)
    testarr_f[num_ch//2]=1.0
    testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
    testarr_w = (np.fft.fft(testarr*window))
    renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    f_len = 220
    num_sample = 100000
    test_std = np.fft.fft(np.random.normal(0,sigma_ch[:,None]/np.sqrt(2),(f_len,num_sample))*window[:,None]*np.sqrt(renorm)+1j*np.random.normal(0,sigma_ch[:,None]/np.sqrt(2),(f_len,num_sample))*window[:,None]*np.sqrt(renorm),axis=0).std()
    sigma_nf = test_std*delta_ch
    data_If = np.fft.fft(data_I*window[:,None],axis=0)*delta_ch*np.sqrt(renorm)
    data_If = np.fft.fftshift(data_If,axes=0)
    data_Vf = np.fft.fft(data_V*window[:,None],axis=0)*delta_ch*np.sqrt(renorm)
    data_Vf = np.fft.fftshift(data_Vf,axes=0)
    sigma_Iindx = np.abs(data_If)>(5*sigma_nf)
    sigma_Vindx = np.abs(data_Vf)>(5*sigma_nf)
    u0_Iindx = np.abs(uarr[indx_I])<100
    v0_Iindx = np.abs(varr[indx_I])<100
    umode = np.sqrt(uarr**2+varr**2)
    eta_h = 2*umode/(freq_arr[0]+freq_arr[-1])
    hor_diff = ((np.abs(eta_arr[:,None])-eta_h[indx_I][None,:])
                /np.abs(eta_h[indx_I][None,:]))
    hor_Iindx = (hor_diff>=hor_low)*(hor_diff<=hor_high)
    #hor_Iindx = (np.abs((np.abs(eta_arr[:,None])-eta_h[indx_I][None,:])/np.abs(eta_h[indx_I][None,:]))<0.3)
    data_rfi = data_I[:,((hor_Iindx*sigma_Iindx).sum(axis=0)>0)]
    time_rfi = timearr[indx_I][((hor_Iindx*sigma_Iindx).sum(axis=0)>0)]
    ifr_rfi = ifr_arr[indx_I][((hor_Iindx*sigma_Iindx).sum(axis=0)>0)]
    sigma_rfi = sigma_Iindx[:,((hor_Iindx*sigma_Iindx).sum(axis=0)>0)]
    frac_rfi = sigma_Iindx[:,u0_Iindx].mean(axis=-1)
    delay_indx = np.where(sigma_rfi.sum(axis=-1)>0)[0]
    delay_sum = np.zeros_like(delay_indx)
    for i in range(len(delay_indx)):
        delay_sum[i] = (hor_Iindx*sigma_Iindx)[delay_indx[i]].sum()
    delay_hor = delay_indx[delay_sum>0]
    uarr_rfi = uarr[indx_I][(hor_Iindx*sigma_Iindx).sum(axis=0)>0]
    varr_rfi = varr[indx_I][(hor_Iindx*sigma_Iindx).sum(axis=0)>0]
    frac_u0 = sigma_Iindx[:,u0_Iindx].mean(axis=-1)
    frac_v0 = sigma_Iindx[:,v0_Iindx].mean(axis=-1)
    sum_dic = ({"block_id":block_id[block_indx],
               "scan_id":scan_id[scan_indx],
               "time_step":time_step,
                "V_avg":V_avg,
                "sigma_ch":sigma_ch,
                "frac_u0":frac_u0,
                "frac_v0":frac_v0,
                "rfi_sum":sigma_Iindx[hor_Iindx].sum(),
                "hor_sum":hor_Iindx.sum(),
                "data_rfi":data_rfi,
                "time_rfi":time_rfi,
                "delay_indx":delay_indx,
                "delay_hor":delay_hor,
                "sigma_rfi":sigma_rfi,
                "ifr_rfi":ifr_rfi,
                "uarr_rfi":uarr_rfi,
                "varr_rfi":varr_rfi,
               })
    for d_i in delay_hor:
        sum_dic.update({"ifr_d_"+str(d_i):ifr_arr[indx_I][(hor_Iindx*sigma_Iindx)[d_i]]})
    with open(save_dir+'/'+str(block_indx*100+scan_indx)+'.pkl', 'wb') as file:
        pickle.dump(sum_dic, file)
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return 1
        

def find_block_id(filename):
    reex = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
    result = re.findall(reex,filename)
    # make sure there is only one block_id in the path
    assert result.count(result[0]) == len(result)
    result = result[0]
    return result

vfind_id = np.vectorize(find_block_id)

def findscan(filename):
    reex = '\.[0-9][0-9][0-9][0-9]\.'
    result = (re.findall(reex, filename)[0])
    return result[1:-1]

vfindscan = np.vectorize(findscan)

def worker_rfi(scan_indx,submslist,uvedges,flag_frac,sel_ch,window,hor_low,hor_high,col='corrected_data',verbose=False):
    msset = ms()
    msset.open(submslist[scan_indx])
    msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]]
    num_ch = sel_ch[0]
    msset.close()
    delta_ch = metadata.chanres(spw=0)[0]
    data,uarr,varr,flag,timearr,ifr_arr = read_ms(submslist[scan_indx],[col,'u','v','flag','time','ifr_number'],sel_ch,verbose)
    time_step = np.unique(timearr)
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    sign_I = np.array([1,1])
    sign_V = np.array([-1j,1j])
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    uarr /= lamb_0
    varr /= lamb_0
    flag_I = (flag[0]+flag[-1])>0  # if XX or YY is flagged then I is flagged 
    indx_I = flag_I.mean(axis=0)<flag_frac
    flag_V = (flag[1]+flag[2])>0  # if XX or YY is flagged then I is flagged 
    indx_V = flag_V.mean(axis=0)<flag_frac
    flag_I = 0
    flag_V = 0
    data_I = np.array([data[0][:,indx_I],data[-1][:,indx_I]])
    data_V = np.array([data[1][:,indx_V],data[2][:,indx_V]])
    if verbose:
        print("Inpainting...",datetime.datetime.now().time().strftime("%H:%M:%S"))
    for i,p_indx in enumerate((0,3)): # XX and YY
        freqfill,farr,rowarr = fill_row(flag[p_indx][:,indx_I]) # find the flags
        data_I[i][farr,rowarr] = data[p_indx][:,indx_I][freqfill,rowarr]
    for i,p_indx in enumerate((1,2)): # XY and YX
        freqfill,farr,rowarr = fill_row(flag[p_indx][:,indx_V]) # find the flags
        data_V[i][farr,rowarr] = data[p_indx][:,indx_V][freqfill,rowarr]
    if verbose:
        print("...Finished",datetime.datetime.now().time().strftime("%H:%M:%S"))
    data_I = (data_I[0]*sign_I[0]+data_I[1]*sign_I[1])/2
    data_V = (data_V[0]*sign_V[0]+data_V[1]*sign_V[1])/2
    data = 0
    sigma_ch = np.sqrt((np.abs(data_V)**2).mean(axis=-1))
    if np.isnan(sigma_ch).sum()>0:
        sigma_ch = fill_nan(sigma_ch)
    eta_arr = np.fft.fftfreq(220,d=delta_ch) # in seconds
    eta_arr = np.fft.fftshift(eta_arr)
    # delay transform
    testarr_f = np.zeros(num_ch)
    testarr_f[num_ch//2]=1.0
    testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
    testarr_w = (np.fft.fft(testarr*window))
    renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    f_len = 220
    num_sample = 100000
    test_std = np.fft.fft(
        (np.random.normal(0,sigma_ch[:,None]/np.sqrt(2),(f_len,num_sample))
         *window[:,None]*np.sqrt(renorm)
         +1j*np.random.normal(0,sigma_ch[:,None]/np.sqrt(2),(f_len,num_sample))
         *window[:,None]*np.sqrt(renorm)),axis=0
    ).std()
    sigma_nf = test_std*delta_ch
    data_If = np.fft.fft(data_I*window[:,None],axis=0)*delta_ch*np.sqrt(renorm)
    data_If = np.fft.fftshift(data_If,axes=0)
    sigma_Iindx = np.abs(data_If)>(5*sigma_nf)
    umode = np.sqrt(uarr**2+varr**2)
    eta_h = 2*umode/(freq_arr[0]+freq_arr[-1])
    hor_diff = ((np.abs(eta_arr[:,None])-eta_h[indx_I][None,:])
                /np.abs(eta_h[indx_I][None,:]))
    hor_Iindx = (hor_diff>=hor_low)*(hor_diff<=hor_high)
    #hor_Iindx = (np.abs((np.abs(eta_arr[:,None])-eta_h[indx_I][None,:])/np.abs(eta_h[indx_I][None,:]))<0.4)
    indx_rfi = ((hor_Iindx*sigma_Iindx).sum(axis=0)==0)
    vis_gridded = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    count = np.zeros((len(uvedges)-1,len(uvedges)-1)) 
    for i in range(num_ch):
        vis_real,_,_ = np.histogram2d(uarr[indx_I][indx_rfi],varr[indx_I][indx_rfi],
                                  bins=[uvedges,uvedges],
                                    weights=(data_I[i,indx_rfi].real))
        vis_imag,_,_ = np.histogram2d(uarr[indx_I][indx_rfi],varr[indx_I][indx_rfi],
                                  bins=[uvedges,uvedges],
                                    weights=(data_I[i,indx_rfi].imag))
        vis_gridded[i] += vis_real+1j*vis_imag
    count,_,_ = np.histogram2d(uarr[indx_I][indx_rfi],varr[indx_I][indx_rfi],
                               bins=[uvedges,uvedges],)
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return vis_gridded,count

def fill_nan(x_arr,sigma=2,truncate=4):
    """
    fill the nan elements of an array with gaussian smoothed interpolation.
    """
    x_alt=x_arr.copy()
    x_alt[np.isnan(x_arr)]=0
    x_alt=gaussian_filter(x_alt,sigma=sigma,truncate=truncate)
    x_test=0*x_arr.copy()+1
    x_test[np.isnan(x_arr)]=0
    x_test=gaussian_filter(x_test,sigma=sigma,truncate=truncate)
    x_smooth=x_alt/x_test
    x_arr[np.isnan(x_arr)]=x_smooth[np.isnan(x_arr)]
    return x_arr

def sim_worker(scan_indx,submslist,sp,uvedges,flag_frac,vis_hi_pad_1,vis_hi_pad_2,conv_factor,sel_ch=None,col='corrected_data',verbose=False):
    sign_I = np.array([1,1])
    sign_V = np.array([-1j,1j])
    num_ch = sp.num_channels
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    filename = submslist[scan_indx]
    msset = ms()
    msset.open(filename)
    if sel_ch is not None:
        msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    if sel_ch is not None:
        freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]] # get the frequencies
    else:
        freq_arr = metadata.chanfreqs(0)[:num_ch]
    msset.close()
    assert len(freq_arr) == num_ch
    #get all the data needed
    data,uarr,varr,flag = read_ms(submslist[scan_indx],[col,'u','v','flag'],sel_ch,verbose)
    data = data[:,:num_ch,:]
    flag = flag[:,:num_ch,:]
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    uarr /= lamb_0
    varr /= lamb_0
    delta_ch = sp.deltav_ch
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    flag_I = (flag[0]+flag[-1])>0  # if XX or YY is flagged then I is flagged 
    indx_I = flag_I.mean(axis=0)<flag_frac
    if indx_I.sum()==0:
        return 0.0,0.0,0.0,0.0,0.0,0.0
    flag_V = (flag[1]+flag[2])>0  # if XX or YY is flagged then I is flagged 
    indx_V = flag_V.mean(axis=0)<flag_frac
    #data_I = np.array([data[0][:,indx_I],data[-1][:,indx_I]])
    data_V = np.array([data[1][:,indx_V],data[2][:,indx_V]])
    tmp_V = ((data_V[0]*sign_V[0]+data_V[1]*sign_V[1])/2)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_ch = np.sqrt((np.sum(np.abs(tmp_V)**2*(1-flag_V[:,indx_V]),axis=-1)/np.sum((1-flag_V[:,indx_V]),axis=-1)))
    if np.isnan(sigma_ch).sum()>0:
        freqfill,farr,rowarr = fill_row(np.isnan(sigma_ch)[:,None])
        sigma_ch[farr] = sigma_ch[freqfill]
    kpara_arr = 2*np.pi*sp.eta_arr()/sp.Y_0()
    uarr = uarr[indx_I]
    varr = varr[indx_I]
    kperp_arr = 2*np.pi*np.sqrt(uarr**2+varr**2)/sp.X_0()
    grid_pos = np.array([(uarr[:,None]>=uvedges[None,:]).sum(axis=-1),
                         (varr[:,None]>=uvedges[None,:]).sum(axis=-1)]).T
    data_hi_1 = vis_hi_pad_1[:,grid_pos[:,0],grid_pos[:,1]]
    data_hi_2 = vis_hi_pad_2[:,grid_pos[:,0],grid_pos[:,1]]
    #phi_arr = phicy(kpara_arr,kperp_arr).T
    #amp_f = np.sqrt(phi_arr/conv_factor/delta_ch**2/1e6) 
    #data_hif = (np.random.normal(0,amp_f/np.sqrt(2),amp_f.shape)
    #        +1j*np.random.normal(0,amp_f/np.sqrt(2),amp_f.shape))
    #data_hi_1 = np.fft.ifft(data_hif,axis=0)
    #data_hif = (np.random.normal(0,amp_f/np.sqrt(2),amp_f.shape)
    #        +1j*np.random.normal(0,amp_f/np.sqrt(2),amp_f.shape))
    #data_hi_2 = np.fft.ifft(data_hif,axis=0)
    data_shape = (len(sigma_ch),len(uarr))
    
    data_tn_1 = (np.random.normal(0,sigma_ch[:,None],data_shape)
                 +1j*np.random.normal(0,sigma_ch[:,None],data_shape))
    data_tn_2 = (np.random.normal(0,sigma_ch[:,None],data_shape)
                 +1j*np.random.normal(0,sigma_ch[:,None],data_shape))
    data_tn_f = [data_tn_1.copy(),data_tn_2.copy()]
    data_hi_f = [data_hi_1.copy(),data_hi_2.copy()]
    for i,p_indx in enumerate((0,3)): # XX and YY
        freqfill,farr,rowarr = fill_row(flag[p_indx][:,indx_I])
        data_hi_f[i][farr,rowarr] = data_hi_f[i][freqfill,rowarr]
        data_tn_f[i][farr,rowarr] = data_tn_f[i][freqfill,rowarr]
    data_hi_f = (data_hi_f[0]+data_hi_f[1])/2
    data_tn_f = (data_tn_f[0]+data_tn_f[1])/2
    data_hi_nf = (data_hi_1+data_hi_2)/2
    data_tn_nf = (data_tn_1+data_tn_2)/2
    flag_I = flag_I[:,indx_I]
    vis_hi_nf,count_nf = grid_vis(uarr,varr,uvedges,data_hi_nf,flag_I,False)
    vis_hi_f,count_f = grid_vis(uarr,varr,uvedges,data_hi_f,np.zeros_like(flag_I),True)
    vis_tn_nf,count_nf = grid_vis(uarr,varr,uvedges,data_tn_nf,flag_I,False)
    vis_tn_f,count_f = grid_vis(uarr,varr,uvedges,data_tn_f,np.zeros_like(flag_I),True)
    return vis_hi_nf,vis_hi_f,vis_tn_nf,vis_tn_f,count_nf,count_f

def save_sim(i,args):
    submslist,sp,uvedges,flag_frac,vis_hi_pad_1,vis_hi_pad_2,conv_factor,sel_ch,col,verbose,scratch_dir = args
    scan_id = vfindscan(submslist)
    block_id = vfind_id(submslist)
    if verbose:
        print('Reading block', block_id[i],'Scan', scan_id[i],datetime.datetime.now().time().strftime("%H:%M:%S"))
    vis_hi_nf,vis_hi_f,vis_tn_nf,vis_tn_f,count_nf,count_f = sim_worker(i,submslist,sp,uvedges,flag_frac,vis_hi_pad_1, vis_hi_pad_2,conv_factor,sel_ch=sel_ch,col=col,verbose=verbose)
    np.save(scratch_dir+'vis_hi_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True',vis_hi_f)
    np.save(scratch_dir+'vis_hi_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False',vis_hi_nf)
    np.save(scratch_dir+'vis_tn_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True',vis_tn_f)
    np.save(scratch_dir+'vis_tn_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False',vis_tn_nf)
    np.save(scratch_dir+'count_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False',count_nf)
    np.save(scratch_dir+'count_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True',count_f)
    if verbose:
        print('Block', block_id[i],'Scan', scan_id[i],'finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return 1

def grid_vis(uarr,varr,uvedges,data,flag,fill):
    num_ch = len(data)
    vis_gridded = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    if fill:
        count = np.zeros((len(uvedges)-1,len(uvedges)-1)) 
    else:
        count = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
    for i in range(num_ch):
        vis_real,_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],
                                      weights=(data[i].real)*(1-flag[i]))
        vis_imag,_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],
                                      weights=(data[i].imag)*(1-flag[i]))
        vis_gridded[i] += vis_real+1j*vis_imag
        if fill:
            continue
        else:
            count[i],_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],weights=(1-flag[i]))
    if fill:
        # if fill, flags are consistent across frequencies
        count,_,_ = np.histogram2d(uarr,varr,bins=[uvedges,uvedges],weights=(1-flag[0]))
    return vis_gridded,count


def sim_realization(i,args):
    scratch_dir,submslist,sp,uvedges,flag_frac,vis_hi_dir,num_sim,sel_ch,col,verbose=args
    scan_id = vfindscan(submslist)
    block_id = vfind_id(submslist)
    if verbose:
        print('Reading block', block_id[i],'Scan', scan_id[i],datetime.datetime.now().time().strftime("%H:%M:%S"))
    sign_I = np.array([1,1])
    sign_V = np.array([-1j,1j])
    num_ch = sp.num_channels
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    filename = submslist[i]
    msset = ms()
    msset.open(filename)
    if sel_ch is not None:
        msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    metadata = msset.metadata()
    if sel_ch is not None:
        freq_arr = metadata.chanfreqs(0)[sel_ch[1]:sel_ch[1]+sel_ch[0]] # get the frequencies
    else:
        freq_arr = metadata.chanfreqs(0)[:num_ch]
    msset.close()
    assert len(freq_arr) == num_ch
    #get all the data needed
    data,uarr,varr,flag = read_ms(submslist[i],[col,'u','v','flag'],sel_ch,verbose)
    data = data[:,:num_ch,:]
    flag = flag[:,:num_ch,:]
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    uarr /= lamb_0
    varr /= lamb_0
    delta_ch = sp.deltav_ch
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    flag_I = (flag[0]+flag[-1])>0  # if XX or YY is flagged then I is flagged 
    indx_I = flag_I.mean(axis=0)<flag_frac
    if indx_I.sum()==0:
        print("All baselines flagged for "+block_id[i] +' scan '+scan_id[i])
        for real_id in range(num_sim):
            np.save(scratch_dir+'vis_hi_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True_'+('%03i'%real_id),0.0+0.0j)
            np.save(scratch_dir+'vis_hi_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False_'+('%03i'%real_id),0.0+0.0j)
            np.save(scratch_dir+'vis_tn_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True_'+('%03i'%real_id),0.0+0.0j)
            np.save(scratch_dir+'vis_tn_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False_'+('%03i'%real_id),0.0+0.0j)
            np.save(scratch_dir+'count_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False_'+('%03i'%real_id),0.0)
            np.save(scratch_dir+'count_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True_'+('%03i'%real_id),0.0)
        return 1.0
    flag_V = (flag[1]+flag[2])>0  # if XX or YY is flagged then I is flagged 
    indx_V = flag_V.mean(axis=0)<flag_frac
    #data_I = np.array([data[0][:,indx_I],data[-1][:,indx_I]])
    data_V = np.array([data[1][:,indx_V],data[2][:,indx_V]])
    tmp_V = ((data_V[0]*sign_V[0]+data_V[1]*sign_V[1])/2)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_ch = np.sqrt((np.sum(np.abs(tmp_V)**2*(1-flag_V[:,indx_V]),axis=-1)/np.sum((1-flag_V[:,indx_V]),axis=-1)))
    if np.isnan(sigma_ch).sum()>0:
        freqfill,farr,rowarr = fill_row(np.isnan(sigma_ch)[:,None])
        sigma_ch[farr] = sigma_ch[freqfill]
    kpara_arr = 2*np.pi*sp.eta_arr()/sp.Y_0()
    uarr = uarr[indx_I]
    varr = varr[indx_I]
    kperp_arr = 2*np.pi*np.sqrt(uarr**2+varr**2)/sp.X_0()
    grid_pos = np.array([(uarr[:,None]>=uvedges[None,:]).sum(axis=-1),
                         (varr[:,None]>=uvedges[None,:]).sum(axis=-1)]).T
    freqfill_XX,farr_XX,rowarr_XX = fill_row(flag[0][:,indx_I])
    freqfill_YY,farr_YY,rowarr_YY = fill_row(flag[3][:,indx_I])
    data_shape = (len(sigma_ch),len(uarr))
    flag_I = flag_I[:,indx_I]
    for real_id in range(num_sim):
        #print(real_id,datetime.datetime.now().time().strftime("%H:%M:%S"))
        vis_hi_pad_1 = np.load(vis_hi_dir+'vishipad1_'+('%03i'%real_id)+'.npy')
        vis_hi_pad_2 = np.load(vis_hi_dir+'vishipad2_'+('%03i'%real_id)+'.npy')
        data_hi_1 = vis_hi_pad_1[:,grid_pos[:,0],grid_pos[:,1]]
        data_hi_2 = vis_hi_pad_2[:,grid_pos[:,0],grid_pos[:,1]]
        data_tn_1 = (np.random.normal(0,sigma_ch[:,None],data_shape)
                 +1j*np.random.normal(0,sigma_ch[:,None],data_shape))
        data_tn_2 = (np.random.normal(0,sigma_ch[:,None],data_shape)
                 +1j*np.random.normal(0,sigma_ch[:,None],data_shape))
        data_tn_f = [data_tn_1.copy(),data_tn_2.copy()]
        data_hi_f = [data_hi_1.copy(),data_hi_2.copy()]
    #for i,p_indx in enumerate((0,3)): # XX and YY
        #freqfill,farr,rowarr = fill_row(flag[p_indx][:,indx_I])
        data_hi_f[0][farr_XX,rowarr_XX] = data_hi_f[0][freqfill_XX,rowarr_XX]
        data_hi_f[1][farr_YY,rowarr_YY] = data_hi_f[1][freqfill_YY,rowarr_YY]
        data_tn_f[0][farr_XX,rowarr_XX] = data_tn_f[0][freqfill_XX,rowarr_XX]
        data_tn_f[1][farr_YY,rowarr_YY] = data_tn_f[1][freqfill_YY,rowarr_YY]
        data_hi_f = (data_hi_f[0]+data_hi_f[1])/2
        data_tn_f = (data_tn_f[0]+data_tn_f[1])/2
        data_hi_nf = (data_hi_1+data_hi_2)/2
        data_tn_nf = (data_tn_1+data_tn_2)/2
        vis_hi_nf,count_nf = grid_vis(uarr,varr,uvedges,data_hi_nf,flag_I,False)
        vis_hi_f,count_f = grid_vis(uarr,varr,uvedges,data_hi_f,np.zeros_like(flag_I),True)
        vis_tn_nf,count_nf = grid_vis(uarr,varr,uvedges,data_tn_nf,flag_I,False)
        vis_tn_f,count_f = grid_vis(uarr,varr,uvedges,data_tn_f,np.zeros_like(flag_I),True)
        np.save(scratch_dir+'vis_hi_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True_'+('%03i'%real_id),vis_hi_f)
        np.save(scratch_dir+'vis_hi_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False_'+('%03i'%real_id),vis_hi_nf)
        np.save(scratch_dir+'vis_tn_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True_'+('%03i'%real_id),vis_tn_f)
        np.save(scratch_dir+'vis_tn_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False_'+('%03i'%real_id),vis_tn_nf)
        np.save(scratch_dir+'count_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_False_'+('%03i'%real_id),count_nf)
        np.save(scratch_dir+'count_'+block_id[i]+'_'+scan_id[i]+'_'+'fill_True_'+('%03i'%real_id),count_f)
    if verbose:
        print('Finished Block', block_id[i],'Scan', scan_id[i],datetime.datetime.now().time().strftime("%H:%M:%S"))
    return 1

def sum_scan_sim(real_id,args):
    num_ch,uvedges,scan_id,scratch_dir,save_dir,block=args
    vis_hi_even_f = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    vis_hi_odd_f = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    vis_tn_even_f = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    vis_tn_odd_f = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    vis_hi_even_nf = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    vis_hi_odd_nf = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    vis_tn_even_nf = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    vis_tn_odd_nf = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    count_even_f = np.zeros((len(uvedges)-1,len(uvedges)-1))
    count_odd_f = np.zeros((len(uvedges)-1,len(uvedges)-1))
    count_even_nf = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
    count_odd_nf = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
    for scan in scan_id:
        vis_hi_i_f = np.load(scratch_dir+'vis_hi_'+block+'_'+scan+'_fill_True_'+('%03i'%real_id)+'.npy')
        vis_hi_i_nf = np.load(scratch_dir+'vis_hi_'+block+'_'+scan+'_fill_False_'+('%03i'%real_id)+'.npy')
        vis_tn_i_f = np.load(scratch_dir+'vis_tn_'+block+'_'+scan+'_fill_True_'+('%03i'%real_id)+'.npy')
        vis_tn_i_nf = np.load(scratch_dir+'vis_tn_'+block+'_'+scan+'_fill_False_'+('%03i'%real_id)+'.npy')
        count_i_f = np.load(scratch_dir+'count_'+block+'_'+scan+'_fill_True_'+('%03i'%real_id)+'.npy')
        count_i_nf = np.load(scratch_dir+'count_'+block+'_'+scan+'_fill_False_'+('%03i'%real_id)+'.npy')
        if int(scan)%2 == 0:
            vis_hi_even_f+=vis_hi_i_f
            vis_hi_even_nf+=vis_hi_i_nf
            vis_tn_even_f+=vis_tn_i_f
            vis_tn_even_nf+=vis_tn_i_nf
            count_even_f+=count_i_f
            count_even_nf+=count_i_nf
        else:
            vis_hi_odd_f+=vis_hi_i_f
            vis_hi_odd_nf+=vis_hi_i_nf
            vis_tn_odd_f+=vis_tn_i_f
            vis_tn_odd_nf+=vis_tn_i_nf
            count_odd_f+=count_i_f
            count_odd_nf+=count_i_nf
    np.save(save_dir+('%03i'%real_id)+'/'+'vissum_hi_'+block+'_fill_True_odd',vis_hi_odd_f)
    np.save(save_dir+('%03i'%real_id)+'/'+'vissum_hi_'+block+'_fill_True_even',vis_hi_even_f)
    np.save(save_dir+('%03i'%real_id)+'/'+'vissum_hi_'+block+'_fill_False_odd',vis_hi_odd_nf)
    np.save(save_dir+('%03i'%real_id)+'/'+'vissum_hi_'+block+'_fill_False_even',vis_hi_even_nf)
    np.save(save_dir+('%03i'%real_id)+'/'+'vissum_tn_'+block+'_fill_True_odd',vis_tn_odd_f)
    np.save(save_dir+('%03i'%real_id)+'/'+'vissum_tn_'+block+'_fill_True_even',vis_tn_even_f)
    np.save(save_dir+('%03i'%real_id)+'/'+'vissum_tn_'+block+'_fill_False_odd',vis_tn_odd_nf)
    np.save(save_dir+('%03i'%real_id)+'/'+'vissum_tn_'+block+'_fill_False_even',vis_tn_even_nf)
    
    np.save(save_dir+('%03i'%real_id)+'/'+'count_'+block+'_fill_True_odd',count_odd_f)
    np.save(save_dir+('%03i'%real_id)+'/'+'count_'+block+'_fill_True_even',count_even_f)
    np.save(save_dir+('%03i'%real_id)+'/'+'count_'+block+'_fill_False_odd',count_odd_nf)
    np.save(save_dir+('%03i'%real_id)+'/'+'count_'+block+'_fill_False_even',count_even_nf)
    return 1


def worker_grid(ms_indx,submslist,uvedges,flag_frac,delay_sigma=5,sel_ch=None,col='corrected_data',verbose=False,save=False,scratch_dir='./',start_ch=0,num_ch=None,taper=blackmanharris):
    sign_I = np.array([1,1])
    sign_V = np.array([-1j,1j])
    scan_id = vfindscan(submslist)
    block_id = vfind_id(submslist)
    block = block_id[ms_indx]
    scan = scan_id[ms_indx]
    if verbose:
        print('Reading block', block,'Scan',scan,
              datetime.datetime.now().time().strftime("%H:%M:%S"))
    data,fg_data,uarr,varr,flag,timearr,ant1,ant2,axis_info = read_ms(
        submslist[ms_indx],
        [col,'model_data','u','v','flag','time','antenna1','antenna2','axis_info'],
        sel_ch,verbose,False
    )
    freq_arr = axis_info['freq_axis']['chan_freq'].ravel()
    if num_ch is None:
        num_ch = len(freq_arr)
    freq_arr = freq_arr[start_ch:(start_ch+num_ch)]
    data = data[:,start_ch:(start_ch+num_ch),:]
    fg_data = fg_data[:,start_ch:(start_ch+num_ch),:]
    flag = flag[:,start_ch:(start_ch+num_ch),:]
    time_step = np.unique(timearr)
    ant_id = np.unique(np.append(ant1,ant2))
    assert (ant_id.size*(ant_id.size-1)/2)*time_step.size == flag.shape[-1]
    ifr_num = ant1*1000+ant2
    ch_arr = np.linspace(0,num_ch-1,num_ch).astype('int')
    dt_min = np.diff(time_step[np.argsort(time_step)]).min()
    sort_indx,tshape,antshape = sort_ifr(timearr,ant1,ant2)
    timearr = timearr[sort_indx].reshape((tshape,antshape))
    uarr = uarr[sort_indx].reshape((tshape,antshape))
    varr = varr[sort_indx].reshape((tshape,antshape))
    flag = flag[:,:,sort_indx].reshape((len(flag),num_ch,tshape,antshape))
    data = data[:,:,sort_indx].reshape((len(data),num_ch,tshape,antshape))
    fg_data = fg_data[:,:,sort_indx].reshape((len(fg_data),num_ch,tshape,antshape))
    ant1 = ant1[sort_indx].reshape((tshape,antshape))
    ant2 = ant2[sort_indx].reshape((tshape,antshape))
    z_0 = 2*f_21/(freq_arr[0]+freq_arr[-1])-1
    lamb_0 = (constants.c/f_21/units.Hz).to('m').value*(1+z_0)
    uarr /= lamb_0
    varr /= lamb_0
    flag_I = (flag[0]+flag[-1])>0 
    indx_I = flag_I.mean(axis=0)<flag_frac
    if indx_I.sum()==0:
        if verbose:
            print('block', block,'Scan',scan,'fully flagged',
              datetime.datetime.now().time().strftime("%H:%M:%S"))
        if save:
            np.save(scratch_dir+'vis_'+block+'_'+scan,0.0+0.0j)
            np.save(scratch_dir+'count_'+block+'_'+scan,0.0)
        return 0.0+0.0j,0.0
    flag_V = (flag[1]+flag[2])>0  # if XX or YY is flagged then I is flagged 
    indx_V = flag_V.mean(axis=0)<flag_frac
    if verbose:
        print('Inpainting...',
              datetime.datetime.now().time().strftime("%H:%M:%S"))
    for i,p_indx in enumerate((0,3)): # XX and YY
        freqfill,farr,rowarr = fill_row(flag[p_indx][:,indx_I]) # find the flags
        data[p_indx][:,indx_I][farr,rowarr] = data[p_indx][:,indx_I][freqfill,rowarr]
        fg_data[p_indx][:,indx_I][farr,rowarr] = fg_data[p_indx][:,indx_I][freqfill,rowarr]
    for i,p_indx in enumerate((1,2)): # XY and YX
        freqfill,farr,rowarr = fill_row(flag[p_indx][:,indx_V]) # find the flags
        data[p_indx][:,indx_V][farr,rowarr] = data[p_indx][:,indx_V][freqfill,rowarr]
    data_I = ((data[0]*sign_I[0]+data[-1]*sign_I[1])/2)
    data_V = ((data[1]*sign_V[0]+data[2]*sign_V[1])/2)
    fg_I = ((fg_data[0]*sign_I[0]+fg_data[-1]*sign_I[1])/2)
    del data,fg_data
    if verbose:
        print('...finished',
              datetime.datetime.now().time().strftime("%H:%M:%S"))
    #sigma_ch = np.std(data_V[:,indx_V],axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_ch = np.sqrt((np.sum(np.abs(data_V[:,indx_V])**2*(1-flag_V[:,indx_V]),axis=-1)/np.sum((1-flag_V[:,indx_V]),axis=-1)))
    if np.isnan(sigma_ch).sum()>0:
        freqfill,farr,rowarr = fill_row(np.isnan(sigma_ch)[:,None])
        sigma_ch[farr] = sigma_ch[freqfill]
    window = taper(num_ch)
    delta_ch = np.diff(freq_arr).mean()
    eta_arr = np.fft.fftfreq(num_ch,d=delta_ch) # in seconds
    eta_arr = np.fft.fftshift(eta_arr)
    # delay transform
    testarr_f = np.zeros(num_ch)
    testarr_f[num_ch//2]=1.0
    testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
    testarr_w = (np.fft.fft(testarr*window))
    renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    f_len = num_ch
    num_sample = 100000
    test_std = np.fft.fft(
        (np.random.normal(0,sigma_ch[:,None]/np.sqrt(2),(f_len,num_sample))
         *window[:,None]*np.sqrt(renorm)
         +1j*np.random.normal(0,sigma_ch[:,None]/np.sqrt(2),(f_len,num_sample))
         *window[:,None]*np.sqrt(renorm)),axis=0
    ).std()
    sigma_nf = test_std*delta_ch
    data_If = np.fft.fft(data_I*window[:,None,None],axis=0)*delta_ch*np.sqrt(renorm)
    data_If = np.fft.fftshift(data_If,axes=0)
    fg_If = np.fft.fft(fg_I*window[:,None,None],axis=0)*delta_ch*np.sqrt(renorm)
    fg_If = np.fft.fftshift(fg_If,axes=0)
    weight_If = np.zeros(data_If.shape)
    weight_If[:,indx_I] = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        cov_diag_data = ((np.abs(data_If*weight_If)**2).sum(axis=1)/(np.abs(weight_If)**2).sum(axis=1)).T
        cov_diag_fg = ((np.abs(fg_If*weight_If)**2).sum(axis=1)/(np.abs(weight_If)**2).sum(axis=1)).T
    
    delay_cut = np.zeros(len(cov_diag_data))
    for ant_indx in range(len(cov_diag_data)):
        pars = fmin(chisq,(1,0.05,0.07),
                    args=(fitfunc,
                          eta_arr*1e6,
                          cov_diag_fg[ant_indx]/cov_diag_fg[ant_indx].max(),
                          1),disp=False,full_output=True
                   )
        if pars[-1]==0:
            pars = pars[0]
            delay_cut[ant_indx] = np.log(100)**(1/pars[0])*pars[1]/1e6
        else:
            pars = pars[0]
            delay_cut[ant_indx] = 0.1/1e6
    delay_indx = np.abs(eta_arr)[None,:]>delay_cut[:,None]
    with np.errstate(divide='ignore', invalid='ignore'):
        delay_flag = (((cov_diag_data-sigma_nf**2)>(
            5*sigma_nf**2
            *np.sqrt(2/time_step.size/indx_I.mean(axis=0))
        )[:,None])
                      *delay_indx)
    bl_flag = delay_flag.sum(axis=-1)>0
    flag_I = np.ones(data_I.shape)
    flag_I[:,(indx_I*(1-bl_flag[None,:])).astype('bool')] = 0
    vis,count = grid_vis(uarr.ravel(),varr.ravel(),uvedges,data_I.reshape((num_ch,-1)),flag_I.reshape((num_ch,-1)),True)
    if save:
        np.save(scratch_dir+'vis_'+block+'_'+scan,vis)
        np.save(scratch_dir+'count_'+block+'_'+scan,count)
    if verbose:
        print('Finished block', block,'Scan',scan,
              datetime.datetime.now().time().strftime("%H:%M:%S"))
    return vis,count