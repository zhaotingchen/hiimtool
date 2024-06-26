'''
The common utility files that does not require any installation of `torch` or `casa`.
'''
import numpy as np
from astropy import constants,units
import re
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.special import erf
from numpy.random import default_rng
from astropy.coordinates import SkyCoord

f_21 = 1420405751.7667 # in Hz

def chisq(pars,func,xarr,yarr,yerr):
    #fitfunc,xarr,yarr,yerr = args
    fitarr = func(xarr,pars)
    return 0.5*np.sum((fitarr-yarr)**2/yerr**2)

def chisq_weighted(data,model,err,weights=None,axis=None):
    """
    Calculate the chisq with custom weights
    """
    if weights is None:
        weights = np.ones_like(err)
    result = ((((data-model)/err)**2*weights).sum(axis=axis)
              /(weights).sum(axis=axis))
    return result

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

def find_scan(filename,strict=False):
    if strict:
        reex = '\.[0-9][0-9][0-9][0-9]\.ms'
    else:
        reex = '\.[0-9][0-9][0-9][0-9]\.'
    result = re.findall(reex, filename)
    if len(result) == 0 and not strict:
        reex = '_[0-9][0-9][0-9][0-9]_'
        result = re.findall(reex, filename)
    if len(result) == 0:
        raise ValueError("no scan id from filename "+filename)
    if result.count(result[0]) != len(result):
        raise ValueError("ambiguous scan id from filename "+filename)
    result = result[0]
    if strict:
        return result[1:-3]
    else:
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

def get_taper_renorm(window):
    N = len(window)
    testarr_f = np.zeros(N)
    testarr_f[N//2]=1.0
    testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
    testarr_w = (np.fft.fft(testarr*window))
    renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    return renorm
    

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
    renorm = get_taper_renorm(window)
    window = np.broadcast_to(window,vis.shape)
    vis_f = (np.fft.fft(vis*window,axis=0)*delta_ch
             *np.sqrt(renorm))
    return vis_f

def get_conv_mat(inp):
    '''
    Generate the matrix representation of a convolution kernel.
    Currently only support 1-D convolutions of two arrays of same length.
    '''
    num_grid = len(inp)
    indx_arr = np.linspace(np.linspace(0,num_grid-1,num_grid),np.linspace(-num_grid+1,0,num_grid),num_grid).T
    window_pad = np.zeros(num_grid*2-1).astype(inp.dtype)
    window_pad[(num_grid-1)//2:((num_grid-1)//2+num_grid)] = inp
    indx_arr = (indx_arr+num_grid-2+num_grid%2).astype('int')
    return window_pad[indx_arr]


def himf(m,phi_s,m_s,alpha_s):
    """
    Analytical HIMF function (or any other Schechter function).
    
    .. math:: \phi = {\rm log}_{10} \phi_* *(m/m_*)^(\alpha_*+1)*e^{-m/m_*}
    
    While the units are arbitrary, it is recommended that 
    phi_s is in the unit of Mpc:sup:`-3`dex:sup:`-1`,
    m_s is in the unit of M_sun,
    alpha_s has no unit.
    
    
    Parameters
    ----------
        m: float array. 
            mass
        phi_s: float.
            HIMF amplitude
        m_s: float.
            knee mass
        alpha_s: float.
            slope
            
    Returns
    -------
        out: float array.
            The HIMF values at m
    """
    out = (np.log(10)*phi_s*(m/m_s)**(alpha_s+1)*np.exp(-m/m_s))
    return out

def himf_pars_jones18(h_70):
    '''
    The HIMF parameters measured in [Jones+18](https://arxiv.org/abs/1802.00053).
    
    Parameters
    ----------
        h_70: float.
            The Hubble parameter over 70 km/s/Mpc.
            
    Returns
    -------
        phi_star: float.
            The amplitude of HIMF in Mpc-3 dex-1.
        
        m_star: float.
            The knee mass of HIMF in log10 solar mass.
            
        alpha: float.
            The slope of HIMF.
    '''
    phi_star = 4.5*1e-3*h_70**3 # in Mpc-3 dex-1
    m_star = np.log10(10**(9.94)/h_70**2)  # in log10 Msun
    alpha = -1.25
    return phi_star,m_star,alpha

def cal_himf(x,mmin,cosmo,mmax=11):
    '''
    Calculate the integrated quantity related to the HIMF.
    
    Parameters
    ----------
        x: list of float. 
            need to be [phi_s,log10(m_s),alpha_s]
        mmin: float.
            The minimum mass to integrate from in log10
        cosmo: an :obj:`astropy.cosmology.Cosmology` object.
            The cosmology object to calculate critical density
        mmax: Optional float, default 11.
            The maximum mass to integrate to in log10.
            
    Returns
    -------
        nhi: float.
            The number density of HI galaxies, in the units of phi_s * dex
        omegahi: float.
            The HI density over the critical density of the present day (assuming the recommended units for x are used)
        psn: float.
            The shot noise in the units of Mpc:sup:`3` (assuming the recommended units for x are used)
    '''
    marr = np.logspace(mmin,mmax,num=500)
    omegahi = (np.trapz(himf(marr,x[0],10**x[1],x[2])*marr,
                        x=np.log10(marr))*units.M_sun/units.Mpc**3
               /cosmo.critical_density0).to('').value
    psn = (np.trapz(himf(marr,x[0],10**x[1],x[2])*marr**2,x=np.log10(marr))
           /np.trapz(himf(marr,x[0],10**x[1],x[2])*marr,x=np.log10(marr))**2
          )
    nhi = np.trapz(himf(marr,x[0],10**x[1],x[2]),x=np.log10(marr))
    return nhi,omegahi,psn

def cumu_nhi_from_himf(m,mmin,x):
    """
    The integrated source number density from HIMF.
    
    Parameters
    ----------
        m: float array.
            The higher end of integration in log10
        mmin: float.
            The minimum mass to integrate from in log10
        x: list of float. 
            need to be [phi_s,log10(m_s),alpha_s]
            
    Returns
    -------
        nhi: float array.
            The integrated number density of HI galaxies, in the units of phi_s * dex
    """
    marr = np.logspace(mmin,m,num=500)
    nhi = np.trapz(himf(marr,x[0],10**x[1],x[2]),x=np.log10(marr),axis=0)
    return nhi

def sample_from_dist(func,xmin,xmax,size=1,cdf=False,seed=None):
    """
    Sample from custom distribution.
    
    Parameters
    ----------
        func: distribution function.
            The probability distribution function (cdf=False) or the cumulative distribution function (cdf=True).
        xmin: float.
            The minimum value to sample from
        xmax: float. 
            The maximum value to sample from
        size: int or list of int, default 1.
            The size of the sample array
        cdf: bool, default False.
            Wheter PDF or CDF is used.
        seed: int, default None.
            Seed for the random number generator. Fixing for reproducible samples.
            
    Returns
    -------
        sample: float array.
            The random sample following the input distribution function.
    """
    xarr = np.linspace(xmin,xmax,1001)
    if cdf is False:
        pdf_arr = func(xarr)
        cdf_arr = np.cumsum(pdf_arr)
    else:
        cdf_arr = func(xarr)
    cdf_arr -= cdf_arr[0]
    cdf_arr /= cdf_arr[-1]
    cdf_inv = interp1d(cdf_arr,xarr)
    rng = default_rng(seed=seed)
    sample = cdf_inv(rng.uniform(low=0,high=1,size=size))
    return sample

def busy_function_simple(xarr,par_a,par_b,par_c,width):
    """
    The simplified busy function that assumes mirror symmetry around x=0 [1].
    
   
    .. math:: B_2(x) = \frac{a}{2} \times ({\rm erf}[b(w^2-x^2)]+1) \times (cx^2+1)
    
    
    Parameters
    ----------
        xarr: float array. 
            the input x values
        par_a: float.
            amplitude parameter
        par_b: float.
            b parameter that controls the sharpness of the double peaks
        par_c: float.
            c parameter that controls the height of the double peaks
        width: float.
            the width of the profile
            
    Returns
    -------
        b2x: float array.
            the busy function values at xarr
            
    References
    ----------
    .. [1] Westmeier, T. et al., "The busy function: a new analytic function for describing the integrated 21-cm spectral profile of galaxies",
           https://ui.adsabs.harvard.edu/abs/arXiv:1311.5308 .
    """
    b2x = (par_a/2*(erf(par_b*(width**2-xarr**2))+1)*(par_c*xarr**2+1))
    return b2x



def dft_mat(num_ch):
    """
    The discrete Fourier transformation matrix.
    
    .. math:: F_{ab} = {\rm exp}[-2\pi i a*b/N ]
    
    Parameters
    ----------
        num_ch: int. 
            the length of the axis to be transformed
            
    Returns
    -------
        kernel: complex array.
            the DFT matrix
    """
    indx_arr = np.linspace(0,num_ch-1,num_ch)
    kernel = np.exp(-2*np.pi*1j*indx_arr[:,None]*indx_arr[None,:]/num_ch)
    return kernel

 
def busy_function_0(xarr,par_a,par_b,par_c,width):
    """
    The symmetric busy function [1].
    
   
    .. math:: B_0(x) = \frac{a}{4} \times ({\rm erf}[b(w-x)]+1) \times ({\rm erf}[b(w+x)]+1) \times (cx^2+1)
    
    
    Parameters
    ----------
        xarr: float array. 
            the input x values
        par_a: float.
            amplitude parameter
        par_b: float.
            b parameter that controls the sharpness of the double peaks
        par_c: float.
            c parameter that controls the height of the double peaks
        width: float.
            the width of the profile
            
    Returns
    -------
        b0x: float array.
            the busy function values at xarr
            
    References
    ----------
    .. [1] Westmeier, T. et al., "The busy function: a new analytic function for describing the integrated 21-cm spectral profile of galaxies",
           https://ui.adsabs.harvard.edu/abs/arXiv:1311.5308 .
    """
    b0x = (par_a/4*(erf(par_b*(width-xarr))+1)*(erf(par_b*(width+xarr))+1)*(par_c*xarr**2+1))
    return b0x

def flux_model(nunu0,iref,coeffs,log=False):
    '''
    flux model using polynomial or log-polynomial
    
    Parameters
    ----------
        nunu0: float array. 
            the input frequency over the reference frequency
        iref: float.
            the flux at the reference frequency
        coeffs: float array.
            the coefficients of the (log-)polynomial. 0th term first.
        log: bool, default False.
            whether to use log-polynomial or polynomial.
            
    Returns
    -------
        ifreq: float array.
            the fluxes at the given frequencies
    '''
    if log:
        exponent = np.sum([coeff*(np.log(nunu0)**(power))
                       for power,coeff in enumerate(coeffs)],axis=0)
        ifreq = iref*nunu0**exponent
    else:
        xarr = nunu0-1
        polyterms = np.sum([coeff*((xarr)**(power+1))
                       for power,coeff in enumerate(coeffs)],axis=0)
        ifreq = iref+polyterms
    return ifreq


def get_mask_renorm_simple(mask):
    testarr = np.ones_like(mask)
    testarr_f = np.fft.fftn(testarr)
    testarr_w = np.fft.fftn(testarr*mask)
    renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    return renorm

def unravel_list(inp):
    """
    unravel a list to one-dimension. Should work for tuple as well.

    Parameters
    ----------
        inp: iterable.

    Returns
    -------
        out: list.
            
    """
    out = [item for sublist in inp for item in sublist]
    return out

def cal_cov_simple(inp):
    '''
    Calculate the covariance of a data vector.
    
    Parameters
    ----------
        data: numpy array. The first axis must of the number of measurements.

    Returns
    -------
        cov: numpy array.
    '''
    data = inp.copy()
    data = data.reshape((len(data),-1))
    data -= data.mean(axis=-1)[:,None]
    cov = np.mean(data[:,None,:]*data[None,:,:],axis=-1)
    return cov

def get_corr_mat(cov):
    '''
    Calculate the correlation of a covariance matrix.
    
    Parameters
    ----------
        cov: numpy array. The covariance matrix.

    Returns
    -------
        corr: numpy array.
    '''
    corr = cov/np.sqrt(np.diagonal(cov))[:,None]/np.sqrt(np.diagonal(cov))[None,:]
    return corr

def strlist_to_str(inp):
    """
    Unravel a list of strs into one comma-separated string.
    """
    out = ''
    for vals in np.array(inp).ravel():
        out += vals+','
    out = out[:-1]
    return out

def jy2_to_k2(sp,fov):
    '''
    Parameters
    ----------
        sp: :class:`.Specs` object.

        fov: float.
        The effective (usually power-squared beam) field-of-view.

    Returns
    -------
        renorm: float.
        The conversion factor from Jy^2Hz^2 to K^2Mpc^3 for the delay power spectrum.
    '''
    renorm = 1/fov/(sp.deltav_ch*sp.num_channels)**2*sp.X_0()**2*(sp.Y_0()*sp.deltav_ch*sp.num_channels)*(sp.lambda_0()**2*units.m**2/2/constants.k_B)**2*units.Jy**2
    renorm = renorm.to('K^2').value
    return renorm

def centre_to_edges(arr):
    '''
    Extend a monotonic array so that the original array is the middle point of the output array.
    '''
    result = arr.copy()
    dx = np.diff(arr)
    result = np.append(result[:-1]-dx/2,result[-2:]+dx[-2:]/2)
    return result

def cov_visual(cov_mat,ax1=0,ax2=1):
    '''
    A function to visualise the covariance matrix. See Eq. 47 of 1910.09273.

    Parameters
    ----------
        cov_mat: numpy array.
        The covariance matrix.
        
        ax1: int, default 0.
        The first axis to visualise.

        ax2: int, default 1.
        The second axis to visualise.

    Returns
    -------
        maj_ax: float.
        the major axis of the 1-sigma ellipse.

        min_ax: float.
        the minor axis of the 1-sigma ellipse.

        phi: float.
        the rotation angle of the ellipse.
    
    '''
    maj_ax = np.sqrt(
        0.5*(cov_mat[ax1,ax1]+cov_mat[ax2,ax2])
        +np.sqrt(0.25*(cov_mat[ax1,ax1]-cov_mat[ax2,ax2])**2+cov_mat[ax1,ax2]**2)
    )
    min_ax = np.sqrt(
        0.5*(cov_mat[ax1,ax1]+cov_mat[ax2,ax2])
        -np.sqrt(0.25*(cov_mat[ax1,ax1]-cov_mat[ax2,ax2])**2+cov_mat[ax1,ax2]**2)
    )
    if cov_mat[ax1,ax2] == 0:
        if cov_mat[ax1,ax1]>=cov_mat[ax2,ax2]:
            phi =0
        else:
            phi = 180
    else:
        phi = 0.5*np.arctan2(2*cov_mat[ax1,ax2],(cov_mat[ax1,ax1]-cov_mat[ax2,ax2]))*180/np.pi
    return maj_ax,min_ax,phi

def tully_fisher(xarr,slope,zero_point,inv=False):
    '''
    Tully-Fisher relation.
    
    Note that, **regardless of inv**, the slope and zero_point always refer to the Tully-Fisher relation
    and **not the inverse**.
    For example, zero_point is always in the unit of log10 mass.
    
    Parameters
    ----------
        xarr: float array. 
            input velocity if inv=False and mass if inv=True.
        slope: float.
            the slope of Tully-Fisher relation.
        zero_point: float.
            the intercept of Tully-Fisher relation
        inv: bool, default False.
            if True, calculate velocity based on input mass.
            
    Returns
    -------
        out: float array.
            The output mass if inv=False and velocity if inv=True.
    '''
    
    if inv:
        out = 10**((np.log10(xarr)-zero_point)/slope)
    else:
        out = 10**(slope*np.log10(xarr)+zero_point)
    return out

def calcsep(ra0,dec0,ra1,dec1):

    """ Returns angular separation between ra0,dec0 and ra1,dec1 in degrees"""

    c1 = SkyCoord(str(ra0)+'deg',str(dec0)+'deg',frame='fk5')
    c2 = SkyCoord(str(ra1)+'deg',str(dec1)+'deg',frame='fk5')
    sep = c1.separation(c2)
    return sep.value

def find_indx_for_subarr(subarr,arr):
    '''
    Find the indices of the elements of an array in another array.
    
    Parameters
    ----------
        subarr: numpy array. 
            The sub-array to search for. Elements can be repeated.
        arr: numpy array.
            the slope of Tully-Fisher relation.
        zero_point: float.
            the intercept of Tully-Fisher relation
        inv: bool, default False.
            if True, calculate velocity based on input mass.
            
    Returns
    -------
        out: float array.
            The output mass if inv=False and velocity if inv=True.
    '''
    assert np.unique(arr).size == arr.size, 'the larger array must be unique'
    # Actually preform the operation...
    arrsorted = np.argsort(arr)
    subpos = np.searchsorted(arr[arrsorted], subarr)
    indices = arrsorted[subpos]
    return indices

def check_unit_equiv(u1,u2):
    """
    Check if two units are equivelant
    """
    return ((1*u1/u2).si.unit == units.dimensionless_unscaled)

def jy_to_kelvin(val,omega,freq):
    '''
    convert Jy/beam to brightness temperature in Kelvin.
    
    Parameters
    ----------
        val: numpy array. 
            The input value(s) in Jy/beam or Jy/pix
        omega: float.
            beam or pixel area in Steradian.
        freq: float.
            the frequency for conversion in Hz.
            
    Returns
    -------
        result: float array.
            The brightness temperature in Kelvin.
    '''
    freq = freq*units.Hz
    omega = omega*units.sr
    result = (val*units.Jy/omega).to(units.K, equivalencies=units.brightness_temperature(freq)).value
    return result