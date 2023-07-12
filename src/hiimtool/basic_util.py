'''
The common utility files that does not require any installation of `torch` or `casa.
'''
import numpy as np
from astropy import constants,units
import re
from scipy.ndimage import gaussian_filter

f_21 = 1420405751.7667 # in Hz

def chisq(pars,func,xarr,yarr,yerr):
    #fitfunc,xarr,yarr,yerr = args
    fitarr = func(xarr,pars)
    return 0.5*np.sum((fitarr-yarr)**2/yerr**2)

def slicer_vectorized(a,start,end):
    """A function for slicing through numpy arrays with string elements"""
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.frombuffer(b.tobytes(),dtype=(str,end-start))


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

    
def find_block_id(filename):
    reex = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
    result = re.findall(reex,filename)
    if result.count(result[0]) != len(result):
        raise ValueError("ambiguous block id from filename "+filename)
    result = result[0]
    return result

vfind_id = np.vectorize(find_block_id)

def find_scan(filename):
    reex = '\.[0-9][0-9][0-9][0-9]\.'
    result = re.findall(reex, filename)
    if result.count(result[0]) != len(result):
        raise ValueError("ambiguous scan id from filename "+filename)
    result = result[0]
    return result[1:-1]

vfind_scan = np.vectorize(find_scan)


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


def p2dim(p2d):
    '''Average cylindrical array with negative and positive k_para to positive only k_para'''
    num_para = p2d.shape[-1]//2
    p2darr = np.zeros((len(p2d), num_para+1))
    p2darr[:,0] = p2d[:,0]
    if p2d.shape[-1]%2 ==0:
        p2darr[:,1:-1] = (p2d[:,1:num_para]+p2d[:,-1:num_para:-1])/2
        p2darr[:,-1] = p2d[:,num_para]
    else:
        p2darr[:,1:] = (p2d[:,1:num_para+1]+p2d[:,-1:-num_para-1:-1])/2
    return p2darr

def itr_tnsq_avg(in_arr,num_sp,max_it=5,sigma=5,frac_lim = 0.01):
    '''Interatively get the underlying average of thermal noise delay ps excluding the peaks'''
    avg_init = in_arr.mean()
    avg = 0
    for it_id in range(max_it):
        avg_thr = avg_init*(1+sigma*np.sqrt(2/num_sp))
        avg_flag = avg_thr>in_arr
        avg = in_arr[avg_flag].mean()
        if np.abs(1-avg/avg_init)<frac_lim:
            break
        avg_init = avg
    return avg

def delay_transform(vis,delta_ch,window):
    '''
    Perform delay transform along the first axis.
    '''
    num_ch = len(vis)
    assert len(window)==num_ch
    testarr_f = np.zeros(num_ch)
    testarr_f[num_ch//2]=1.0
    testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
    testarr_w = (np.fft.fft(testarr*window))
    renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    vis_f = np.fft.fft(vis,axis=0)*delta_ch
    return vis_f

def get_conv_mat(inp):
    '''
    Generate the matrix representation of a convolution kernel.
    Currently only support 1-D convolutions of two arrays of same length.
    '''
    num_grid = len(inp)
    indx_arr = np.linspace(np.linspace(0,num_grid-1,num_grid),np.linspace(-num_grid+1,0,num_grid),num_grid).T
    window_pad = np.zeros(num_grid*2-1)
    window_pad[(num_grid-1)//2:((num_grid-1)//2+num_grid)] = inp
    indx_arr = (indx_arr+num_grid-2+num_grid%2).astype('int')
    return window_pad[indx_arr]