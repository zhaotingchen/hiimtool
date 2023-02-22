import numpy as np
import time
from astropy.cosmology import Planck18
from .hiimvis import Specs,vis_power_3d


def worker(sim_id,pars):
    """
    A worker function for simulating the thermal noise for one realisation.
    
    Parameters
    ----------
        sim_id: int
            The id number for the realisation.
            
        pars: list
            A list of all the input parameters. See the below list for each one.
        
        sigma_n: float 
            The standard deviation of thermal noise for one baseline at one timestep
        
        shape: list of int
            The shape of the gridded visibility data (n_ch,n_u,n_v)
        
        counti_even: np.array int
            The number of baselines in each u-v grid in each channel for the even scans. If fill=True, array has shape (n_u,n_v). Else array has shape (n_ch,n_u,n_v)
        
        counti_odd: np.array int
            The number of baselines in each u-v grid in each channel for the odd scans. If fill=True, array has shape (n_u,n_v). Else array has shape (n_ch,n_u,n_v)
        
        window: np.array of size (n_ch)
            The frequency taper used in power spectrum estimation.
        
        fill: bool
            Whether the flagged channel is filled by the nearest neighbor.
        
        sp: `hiim.vis.Specs` instance, containing the metadata of the observations.
    
    Returns
    -------
        pd3d_even: array of the 3-D thermal noise power spectrum for the even scan. 
        
        pd3d_odd: array of the 3-D thermal noise power spectrum for the odd scan.
        
        pd3dx: array of the 3-D thermal noise power spectrum for the cross between even and odd scans.
        
        
    """
    sigma_n = pars[0]
    shape = pars[1]
    counti_even = pars[2]
    counti_odd = pars[3]
    window = pars[4]
    fill = pars[5]
    sp = pars[6]
    noise_even = (np.random.normal(0.0,sigma_n/np.sqrt(2),size= shape).astype('complex')
              +np.random.normal(0.0,sigma_n/np.sqrt(2),size= shape).astype('complex')*1j)
    if fill:
        noise_even /= np.sqrt(counti_even)[None,:]
    else:
        noise_even /= np.sqrt(counti_even)
    

    noise_odd = (np.random.normal(0.0,sigma_n/np.sqrt(2),size= shape).astype('complex')
              +np.random.normal(0.0,sigma_n/np.sqrt(2),size= shape).astype('complex')*1j)
    if fill:
        noise_odd /= np.sqrt(counti_odd)[None,:]
    else:
        noise_odd /= np.sqrt(counti_odd)
    

    pd3d_even = vis_power_3d(sp,noise_even,window = window)
    pd3d_odd = vis_power_3d(sp,noise_odd,window = window)
    pd3dx = vis_power_3d(sp,noise_odd,vis_2=noise_even,window = window)
    #np.save('../tnsimpsx_220/pd3d_fill_'+sim_id,pd3dx)
    #np.save('../tnsimps_even_220_fill/pd3d_fill_'+sim_id,pd3d_even)
    #np.save('../tnsimps_odd_220/pd3d_fill_'+sim_id,pd3d_odd)
    return pd3d_even,pd3d_odd,pd3dx

def sumtn(id_num,pars):
    """
    A function to run `num_loop` realistions and average the power spectra and save it as `id_num`. Note that the cross power spectrum average is the average of the absolute value.
    
    Parameters
    ----------
        id_num: int
            The id number for the average.
            
        pars: list
            A list of all the input parameters. See the below list for each one.
        
        sigma_n: float 
            The standard deviation of thermal noise for one baseline at one timestep
        
        shape: list of int
            The shape of the gridded visibility data (n_ch,n_u,n_v)
        
        counti_even: np.array int
            The number of baselines in each u-v grid in each channel for the even scans. If fill=True, array has shape (n_u,n_v). Else array has shape (n_ch,n_u,n_v)
        
        counti_odd: np.array int
            The number of baselines in each u-v grid in each channel for the odd scans. If fill=True, array has shape (n_u,n_v). Else array has shape (n_ch,n_u,n_v)
        
        window: np.array of size (n_ch)
            The frequency taper used in power spectrum estimation.
        
        fill: bool
            Whether the flagged channel is filled by the nearest neighbor.
        
        sp: `hiim.vis.Specs` instance
            containing the metadata of the observations.
        
        scratch_dir: string
            The scratch folder to store the outputs.
        
        num_loop: int
            number of realisations to average across.
    
    Returns
    -------
        1
    """
    num_loop = pars[-1]
    scratch_dir = pars[-2]
    num_bl = pars[2].shape[-1]
    sp = pars[6]
    tn_even = np.zeros((sp.num_channels,num_bl),dtype='complex')
    tn_odd = np.zeros((sp.num_channels,num_bl),dtype='complex')
    tn_x = np.zeros((sp.num_channels,num_bl),dtype='complex')
    for i in range(num_loop):
        tn_even_i,tn_odd_i,tn_x_i = worker(i,pars)
        tn_even += tn_even_i
        tn_odd += tn_odd_i
        tn_x += np.abs(tn_x_i)
    tn_even /= num_loop
    tn_odd /= num_loop
    tn_x /= num_loop
    np.save(scratch_dir+'tn_even_'+str(id_num),tn_even)
    np.save(scratch_dir+'tn_odd_'+str(id_num),tn_odd)
    np.save(scratch_dir+'tn_x_'+str(id_num),tn_x)
    return 1

def sumtn_var(id_num,pars):
    num_loop = pars[-1]
    tn_even_avg = pars[-2]
    tn_odd_avg = pars[-3]
    scratch_dir = pars[-4]
    num_bl = pars[2].shape[-1]
    sp = pars[6]
    tn_even_var = np.zeros((sp.num_channels,num_bl),dtype='complex')
    tn_odd_var = np.zeros((sp.num_channels,num_bl),dtype='complex')
    tn_x_var = np.zeros((sp.num_channels,num_bl),dtype='complex')
    for i in range(num_loop):
        tn_even_i,tn_odd_i,tn_x_i = worker(i,pars[:-4])
        tn_even_var += (tn_even_i-tn_even_avg)**2
        tn_odd_var += (tn_odd_i-tn_odd_avg)**2
        tn_x_var += (tn_x_i)**2
    tn_even_var /= num_loop
    tn_odd_var /= num_loop
    tn_x_var /= num_loop
    np.save(scratch_dir+'tn_even_'+str(id_num),tn_even_var)
    np.save(scratch_dir+'tn_odd_'+str(id_num),tn_odd_var)
    np.save(scratch_dir+'tn_x_'+str(id_num),tn_x_var)
    return 1