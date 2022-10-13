import numpy as np
import matplotlib.pyplot as plt
import time
from astropy.cosmology import Planck18_arXiv_v2 as Planck18
from hiimvis import  Specs,inversemab,vis_power,BrightnessTempPS,vis_power_3d
import glob
import torch
from torch.fft import fft,fftn,fftfreq
from scipy.signal import blackmanharris
from astropy import units,constants
import sys
from mpi4py import MPI

sim_num = int(sys.argv[1])
num_ch = int(sys.argv[2])
sp = Specs(cosmo = Planck18,
           freq_start_hz = 1054535156.25,
           num_channels = num_ch,
           deltav_ch = 208984.375,
           FWHM_ref = 83.94201153142113/60*np.pi/180,
           FWHM_freq_ref = 1050e6,)

ae_over_tsys = 6.22 * units.m**2/units.K
deltat = 8*units.s
deltav = 208984.375*units.Hz
sigma_n = (2*constants.k_B/ae_over_tsys/np.sqrt(deltat*deltav)).to('Jy').value

visavg_odd = np.load('../gridded_vis/visavg_odd_220.npy')
visavg_even = np.load('../gridded_vis/visavg_even_220.npy')
counttot_even = np.load('../gridded_vis/count_even_220.npy')
counttot_odd = np.load('../gridded_vis/count_odd_220.npy')

umodeedges = np.linspace(-6000,6000,201)
ucen = (umodeedges[1:]+umodeedges[:-1])/2
umode_i = np.sqrt(ucen[:,None]**2+ucen[None,:]**2)
indx = np.prod(counttot_even>0,axis=0)*np.prod(counttot_odd>0,axis=0)
#indx *= (1-(ucen>=-30)*(ucen<=30))[:,None] # u=0 mask
counti_even = counttot_even[np.where(np.broadcast_to(indx,visavg_even.shape))].reshape(sp.num_channels,-1)
counti_odd = counttot_odd[np.where(np.broadcast_to(indx,visavg_odd.shape))].reshape(sp.num_channels,-1)
umode_i = umode_i[np.where(indx)]
umodeedges = np.linspace(0,6000,61)

window = blackmanharris(sp.num_channels)

def worker(sim_id):
    noise_even = (np.random.normal(0.0,sigma_n/np.sqrt(2),size= counti_even.shape).astype('complex')
              +np.random.normal(0.0,sigma_n/np.sqrt(2),size= counti_even.shape).astype('complex')*1j)
    noise_even /= np.sqrt(counti_even)
    noise_odd = (np.random.normal(0.0,sigma_n/np.sqrt(2),size= counti_odd.shape).astype('complex')
              +np.random.normal(0.0,sigma_n/np.sqrt(2),size= counti_odd.shape).astype('complex')*1j)
    noise_odd /= np.sqrt(counti_odd)
    pd3d_even = vis_power_3d(sp,noise_even,window = window)
    pd3d_odd = vis_power_3d(sp,noise_odd,window = window)
    np.save('../tnsimps_even_220/pd3d_'+sim_id,pd3d_even)
    np.save('../tnsimps_odd_220/pd3d_'+sim_id,pd3d_odd)
    return 1


rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

for i in range(sim_num):
    if i%size!=rank: continue
    #print ("Task number %d being done by processor %d of %d" % (i,rank, size))
    worker(str(i))