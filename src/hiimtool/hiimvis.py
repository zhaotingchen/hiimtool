import numpy as np
from astropy import constants,units
import time
import warnings
from itertools import product
import psutil
import torch
from torch.fft import fft,fftn,fftfreq
from .basic_util import Specs,f_21,get_taper_renorm
from torch import histogramdd

_range = range

def grid_avg(sp,uedges,visdata,bldata,device=None,verbose=False):
    """Grid the visibility data onto u-v grids"""
    if verbose:
        start_time = time.time()
        print("grid_avg: start gridding visibility")
    # if not specified, do it on cpu
    if device is None:
        device = 'cpu'
    # no need to waster gpu memory here
    bldata = torch.from_numpy(bldata.T)
    lamb_arr = torch.from_numpy(sp.lamb_arr())
    # uedges needs to be passed to histrogram so on gpu
    uedges = torch.from_numpy(uedges).to(device)
    vis_gridded = torch.zeros((sp.num_channels,len(uedges)-1,
                            len(uedges)-1),dtype=torch.complex64)
    count = torch.zeros((sp.num_channels,len(uedges)-1,len(uedges)-1))
    if verbose:
        bar = get_bar()
        irange = bar(range(sp.num_channels))
    else:
        irange = range(sp.num_channels)
        
    for i in irange:
        # from meter to lambda
        sample = (bldata[:,:-1]/lamb_arr[i]).to(device)
        vis_ch = torch.from_numpy(visdata[i]).to(device)
        # counting the number of baselines in each grid
        count_ch = histogramdd(sample.T,bins=[uedges,uedges])[0]
        # sum of visibility
        vis_weight = (histogramdd(sample.T,bins=[uedges,uedges],
                                  weights=vis_ch.real)[0]
                      +1j*histogramdd(sample.T,bins=[uedges,uedges],
                                      weights=vis_ch.imag)[0])
        # averaged gridded visibility
        vis_weight = (vis_weight/count_ch)
        vis_weight[vis_weight != vis_weight] = 0 #nan_to_zero
        vis_gridded[i] = vis_weight.cpu()
        count[i] = count_ch.cpu()
    #take out the zero ones
    vis_gridded = vis_gridded.reshape((sp.num_channels,-1))
    count = count.reshape((sp.num_channels,-1))
    arr = torch.prod(count!=0,dim=0)
    indx = (torch.where(arr!=0)[0])
    u_cen = ((uedges[1:]+uedges[:-1])/2).cpu()
    uu,vv = torch.meshgrid(u_cen,u_cen)
    umode_i = (torch.sqrt(torch.flatten(uu)**2+torch.flatten(vv)**2))
    vis_gridded = (vis_gridded[:,indx]).numpy()
    count = (count[:,indx]).numpy()
    umode_i = (umode_i[indx]).numpy()
    #clear memory
    u_cen = 0
    uu = 0
    vv = 0
    arr = 0
    sample = 0
    vis_ch = 0
    count_ch = 0
    bldata = 0
    lamb_arr = 0
    uedges = 0
    vis_weight = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    if verbose:
        print("gridding finished! Time spent: %f seconds" % 
              (time.time()-start_time))
    return vis_gridded,count,umode_i    
    
def vis_power_3d(sp,vis_gridded,vis_2=None,
              device = None,verbose=False,sigma_cut=np.inf,window=None,
             return_sigma=False):
    """Calculate the 3D delay power spectrum at each (gridded) baseline"""
    if window is None:
        window = np.ones(sp.num_channels)
        renorm = 1.0
    else:
        renorm = get_taper_renorm(window)
        #testarr_f = np.zeros(sp.num_channels)
        #testarr_f[sp.num_channels//2]=1.0
        #testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
        #testarr_w = (np.fft.fft(testarr*window))
        #renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    window = torch.from_numpy(window).to(device)
    if verbose:
        start_time = time.time()
        print("vis_power: start calculating delay power spectrum")
    if device is None:
        device = 'cpu'
    if vis_2 is None:
        vis_2 = vis_gridded
    if device != 'cpu':
        torch.cuda.empty_cache()
    #large array. store in cpu first
    vis_gridded = vis_gridded.reshape((sp.num_channels,-1))
    vis_gridded = torch.from_numpy(vis_gridded).to(device)
    vis_gridded[vis_gridded != vis_gridded] = 0 #nan_to_zero just in case
    vis_gridded = fft(vis_gridded*window[:,None],dim=0)*sp.deltav_ch #Fourier Convention
    vis_2 = vis_2.reshape((sp.num_channels,-1))
    vis_2 = torch.from_numpy(vis_2).to(device)
    vis_2[vis_2 != vis_2] = 0 #nan_to_zero just in case
    vis_2 = fft(vis_2*window[:,None],dim=0)*sp.deltav_ch #Fourier Convention
    vis_gridded = (torch.conj(vis_2)*vis_gridded).real
    #large array. store in cpu first
    pdelay = vis_gridded.to('cpu')*renorm 
    #clear memory
    vis_gridded = 0
    vis_2 = 0
    vis_ch = 0
    pd_ch = 0
    wai = 0
    weight = 0
    pd_avg = 0
    pd_std = 0
    vis_ch_mean = 0
    vis_ch_sigma = 0
    umodeedges = 0
    umode_i = 0
    window = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    if verbose:
        print("delay ps calculation finished! Time spent: %f seconds" % 
              (time.time()-start_time))
    return pdelay.numpy()

def vis_power(sp,umode_i,umodeedges,vis_gridded,vis_2=None,
              device = None,verbose=False,sigma_cut=np.inf,window=None,
             return_sigma=False):
    """Calculate the cylindrical delay power spectrum"""
    if window is None:
        window = np.ones(sp.num_channels)
        renorm = 1.0
    else:
        renorm = get_taper_renorm(window)
        #testarr_f = np.zeros(sp.num_channels)
        #testarr_f[sp.num_channels//2]=1.0
        #testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
        #testarr_w = (np.fft.fft(testarr*window))
        #renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    window = torch.from_numpy(window).to(device)
    if verbose:
        start_time = time.time()
        print("vis_power: start calculating delay power spectrum")
    if device is None:
        device = 'cpu'
    if vis_2 is None:
        vis_2 = vis_gridded
    # |u| of each grid
    umode_i = np.ravel(umode_i)
    indx = np.where((umode_i<umodeedges.max()) & (umode_i>umodeedges.min()))[0]
    umodeedges = torch.from_numpy(umodeedges).to(device)
    # this is large. store in cpu first
    umode_i = torch.from_numpy(umode_i[indx]).to(device)
    #number of baseline in each grid, averaged across the frequency channels
    #large array. store in cpu first
    if device != 'cpu':
        torch.cuda.empty_cache()
    # this is a large array, so clear cache before calculation
    wai = ((umode_i[:,None]>umodeedges[None,:-1])
           *(umode_i[:,None]<umodeedges[None,1:]))
    #large array. store in cpu first
    vis_gridded = vis_gridded.reshape((sp.num_channels,-1))
    vis_gridded = torch.from_numpy(vis_gridded[:,indx]).to(device)
    vis_gridded[vis_gridded != vis_gridded] = 0 #nan_to_zero just in case
    vis_gridded = fft(vis_gridded*window[:,None],dim=0)*sp.deltav_ch #Fourier Convention
    vis_2 = vis_2.reshape((sp.num_channels,-1))
    vis_2 = torch.from_numpy(vis_2[:,indx]).to(device)
    vis_2[vis_2 != vis_2] = 0 #nan_to_zero just in case
    vis_2 = fft(vis_2*window[:,None],dim=0)*sp.deltav_ch #Fourier Convention
    vis_gridded = (torch.conj(vis_2)*vis_gridded).real
    #large array. store in cpu first
    vis_gridded = vis_gridded.to('cpu')
    #check if memory is enough
    if device == 'cpu':
        # if on cpu, the memory is the availabel ram
        total_mem = psutil.virtual_memory().available
    else:
        # if on gpu, the memory is the gpu one
        total_mem = torch.cuda.get_device_properties(
            torch.cuda.current_device()).total_memory
    #array_mem is complex64 in this case
    array_mem = (sp.num_channels*len(indx)*(len(umodeedges)-1)*64/8)*2
    if array_mem>total_mem:
        if verbose:
            print("Array too large! Split into %i chunks" % sp.num_channels)
            irange = range(sp.num_channels)
        else:
            irange = range(sp.num_channels)
        pdelay = np.zeros((sp.num_channels,len(umodeedges)-1),dtype='complex64')
        for i in irange:
            vis_ch = vis_gridded[i,:].to(device)
            vis_ch_mean = torch.sum(vis_ch[:,None]*wai,dim=0)/torch.sum(wai,dim=0) # get the average
            vis_ch_sigma = (torch.sqrt(torch.sum((vis_ch[:,None]-vis_ch_mean[None,:])**2*wai,dim=0)
                                       /torch.sum(wai,dim=0))) # get the std
            weight = wai*(torch.abs(vis_ch[:,None]-vis_ch_mean[None,:])<=sigma_cut*vis_ch_sigma[None,:]) # sigma cut
            pd_ch = (torch.sum(vis_ch[:,None]*weight[:,:],dim=0)
                     /(torch.sum(weight,dim=0)))
            if return_sigma is True:
                pdelay[i] = vis_ch_sigma.cpu().numpy()*renorm
            else:
                pdelay[i] = pd_ch.cpu().numpy()*renorm        
    else:
        vis_gridded = vis_gridded.to(device)
        pd_avg = (torch.sum(vis_gridded[:,:,None]*wai[None,:,:],dim=1)
                  /(torch.sum(wai,dim=0))[None,:])
        pd_std = torch.sqrt((torch.sum((vis_gridded[:,:,None]-pd_avg[:,None,:])**2*wai[None,:,:],dim=1)
                  /(torch.sum(wai,dim=0))[None,:]))
        weight = wai[None,:,:]*(torch.abs(vis_gridded[:,:,None]-pd_avg[:,None,:])<=sigma_cut*pd_std[:,None,:])
        if return_sigma is True:
            pdelay = pd_std.cpu().numpy()*renorm
        else:
            pdelay = (torch.sum(vis_gridded[:,:,None]*weight,dim=1)
                  /(torch.sum(weight,dim=1))).cpu().numpy()*renorm
        
    uarr = ((umodeedges[:-1]+umodeedges[1:])/2).cpu().numpy()
    #pdelay = pdelay
    #clear memory
    vis_gridded = 0
    vis_2 = 0
    vis_ch = 0
    pd_ch = 0
    wai = 0
    weight = 0
    pd_avg = 0
    pd_std = 0
    vis_ch_mean = 0
    vis_ch_sigma = 0
    umodeedges = 0
    umode_i = 0
    window = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    if verbose:
        print("delay ps calculation finished! Time spent: %f seconds" % 
              (time.time()-start_time))
    return pdelay,uarr,sp.eta_arr()

def bin_3d_to_cy(ps3d,umode_i,umodeedges,device='cpu',weights=None):
    ps3d = torch.from_numpy(ps3d).to(device)
    if weights is None:
        weights = torch.ones_like(ps3d)
    else:
        weights = torch.from_numpy(weights).to(device)
    umode_i = torch.from_numpy(umode_i).to(device)
    umodeedges = torch.from_numpy(umodeedges).to(device)
    indx = (umode_i[:,None]>=umodeedges[None,:-1])*(umode_i[:,None]<umodeedges[None,1:])
    pscy = torch.sum(ps3d[:,:,None]*indx[None,:,:]*weights[:,:,None],dim=1)/torch.sum(indx[None,:,:]*weights[:,:,None],dim=1)
    pscy = pscy.cpu().numpy()
    # clear cache
    ps3d = 0
    umode_i = 0 
    umodeedges = 0
    indx = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    return pscy

def bin_3d_to_1d(sp,ps3d,umode_i,k1dedges,device='cpu',weights=None,error=False):
    ps3d = torch.from_numpy(ps3d).to(device)
    if weights is None:
        weights = torch.ones_like(ps3d)
    else:
        weights = torch.from_numpy(weights).to(device)
    umode_i = 2*np.pi*torch.from_numpy(umode_i).to(device)/sp.X_0() # to kperp
    eta_i = torch.from_numpy(2*np.pi*sp.eta_arr()/sp.Y_0()) # to kpara
    kfield = torch.sqrt(eta_i[:,None]**2+umode_i[None,:]**2)
    k1dcen = (k1dedges[1:]+k1dedges[:-1])/2
    k1dedges = torch.from_numpy(k1dedges).to(device)
    indx = (kfield[:,:,None]>=k1dedges[None,None,:-1])*(kfield[:,:,None]<k1dedges[None,None,1:])
    ps1d = torch.sum(ps3d[:,:,None]*indx*weights[:,:,None],dim=(1,0))/torch.sum(indx*weights[:,:,None],dim=(1,0))
    if error is True:
        ps1derr = torch.sqrt(torch.sum((ps3d[:,:,None]-ps1d[None,None,:])**2*(indx*weights[:,:,None])**2,dim=(1,0))/torch.sum((indx*weights[:,:,None]),dim=(1,0))**2)
        ps1derr = ps1derr.cpu().numpy()
    ps1d = ps1d.cpu().numpy()
    # clear cache
    ps3d = 0
    umode_i = 0 
    k1dedges = 0
    indx = 0
    kfield = 0
    eta_i = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    if error is True:
        return ps1d,ps1derr,k1dcen
    else:
        return ps1d,k1dcen


def inversemab(sp,umode_i,umodeedges,
               int_steps = 200, approxlim = 40, 
               kmodeedges=None, device=None,verbose=False):
    """Calculate the conversion matrix"""
    if verbose:
        start_time = time.time()
        print("inversemab: start calculating the conversion matrix")
    if device is None:
        device = 'cpu'
    umode_i = np.ravel(umode_i)
    indx = np.where((umode_i<umodeedges.max()) & (umode_i>umodeedges.min()))[0]
    umode_i = torch.from_numpy(umode_i[indx]).to(device)
    umodeedges = torch.from_numpy(umodeedges).to(device)
    if kmodeedges is None:
        # this is preferred as it gives almost diaganol matrix
        kmodeedges = (2*np.pi*umodeedges/sp.X_0()).cpu().numpy()
    else:
        warnings.warn("Using custom kperp may lead to errors in matrix")
    wai = ((umode_i[:,None]>umodeedges[None,:-1])
           *(umode_i[:,None]<umodeedges[None,1:]))
    # the integration steps for every kbin.
    # This step must come from numpy as torch linspace seems to not support multi layers
    kintegarr = torch.from_numpy(
        np.linspace(kmodeedges[:-1],kmodeedges[1:],int_steps).T)
    # the k of each uv grid
    kmode_i = 2*np.pi*umode_i/sp.X_0()
    umode_i = 0 # save some memory
    # sigma^2 X^2 k k_i
    if device == 'cpu':
        # if on cpu, the memory is the availabel ram
        total_mem = psutil.virtual_memory().available
    else:
        # if on gpu, the memory is the gpu one
        total_mem = torch.cuda.get_device_properties(
            torch.cuda.current_device()).total_memory
    array_mem = (len(kmode_i)*(len(kmodeedges)-1)*int_steps*32/8)*10
    if array_mem > total_mem:
        if verbose:
            print("Array too large! Split into %i chunks" %(len(kmodeedges)-1))
            irange = range((len(kmodeedges)-1))
        else:
            irange = range((len(kmodeedges)-1))
        #store in cpu
        integarr = torch.zeros((len(kmode_i),len(kmodeedges)-1),device='cpu')
        for i in irange:
            kinteg_ch = kintegarr[None,i,:].to(device)
            s2X2kki= sp.sigma_0()**2*sp.X_0()**2*kinteg_ch*kmode_i[:,None]
            indx = torch.where(s2X2kki<approxlim)
            # fill the array with approximated value
            integrand = (torch.exp(-0.5*sp.sigma_0()**2*sp.X_0()**2
                                   *(kinteg_ch-kmode_i[:,None])**2)
                         /torch.sqrt(2*np.pi*s2X2kki))
            # modified Bessel function of the first kind
            i0arr = torch.i0(s2X2kki[indx])
            # exp term
            exparr = torch.exp(
                (-0.5*sp.sigma_0()**2*sp.X_0()**2
                 *(kinteg_ch**2+kmode_i[:,None]**2))[indx])
            integrand[indx] = i0arr*exparr
            integrand *= kinteg_ch*np.pi/2*sp.sigma_0()**4
            integarr[:,i] = torch.trapz(integrand,x=kinteg_ch,dim=-1).cpu()
    else:
        kintegarr = kintegarr.to(device)
        s2X2kki= sp.sigma_0()**2*sp.X_0()**2*kintegarr[None,:,:]*kmode_i[:,None,None]
        indx = torch.where(s2X2kki<approxlim)
        # fill the array with approximated value
        integrand = (torch.exp(-0.5*sp.sigma_0()**2*sp.X_0()**2
                               *(kintegarr[None,:,:]-kmode_i[:,None,None])**2)
                     /torch.sqrt(2*np.pi*s2X2kki))
        # modified Bessel function of the first kind
        i0arr = torch.i0(s2X2kki[indx])
        # exp term
        exparr = torch.exp((-0.5*sp.sigma_0()**2*sp.X_0()**2*
                            (kintegarr[None,:,:]**2+kmode_i[:,None,None]**2))[indx])
        # some low x does not need to be approximated
        integrand[indx] = i0arr*exparr
        # save some RAM
        i0arr = 0 
        exparr = 0
        integrand *= kintegarr[None,:,:]*np.pi/2*sp.sigma_0()**4
        #integrate the beam term
        integarr = torch.trapz(integrand,x=kintegarr[None,:,:],dim=-1)
    #clear memory
    kinteg_ch = 0
    s2X2kki = 0
    integrand = 0
    i0arr = 0
    exparr = 0
    indx = 0 
    if device != 'cpu':
        torch.cuda.empty_cache()
    #print("test")
    # construct the matrix
    array_mem = (len(kmode_i)*(len(kmodeedges)-1)*(len(umodeedges)-1)*32/8)*2
    if array_mem > total_mem:
        if verbose:
            print("Array too large! Split into %i chunks" %(len(kmodeedges)-1))
            irange = range((len(kmodeedges)-1))
        else:
            irange = range((len(kmodeedges)-1))
        mab = torch.zeros((len(umodeedges)-1,len(kmodeedges)-1),device=device)
        for i in irange:
            integ_ch = integarr[:,None,i].to(device)
            mab[:,i] = torch.sum(wai[:,:]*integ_ch
                                 ,dim=0)/torch.sum(wai,dim=0)
    else:
        integarr = integarr.to(device)
        mab = torch.sum(wai[:,:,None]*integarr[:,None,:]
                        ,dim=0)/torch.sum(wai,dim=0)[:,None]

    # add the coefficient term for all the units
    coeff = ((2*constants.k_B/(sp.lambda_0()*units.m)**2)**2*sp.num_channels
             *sp.deltav_ch*units.Hz/(sp.Y_0()*units.Mpc/units.Hz)
             /units.Mpc**2*units.K**2*units.Mpc**3).to("Jy^2Hz^2").value
    mab *= coeff
    inverseM = torch.inverse(mab)
    result = inverseM.cpu().numpy().real
    #clear memory
    mab = 0
    integarr = 0
    integ_ch = 0
    wai = 0
    kmode_i = 0
    umodeedges = 0 
    count = 0
    inverseM = 0
    n_i = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    if verbose:
        print("matrix calculation finished! Time spent: %f seconds" % 
              (time.time()-start_time))
    return result


class BrightnessTempPS():
    """A class for brightness temperature power spectrum calculation"""
    def __init__(self, 
                 sp,
                 vis_gridded,
                 umode_i,
                 umodeedges,
                 vis_2=None,
                 kperpedges=None,
                 k1dedges=None,
                 approxlim=30,
                 int_steps=200,
                 mask=1,
                 device=None,
                 verbose=False,
                 sigma_cut=np.inf,
                 window=None,
                ):
        self.sp = sp
        self.vis_gridded = vis_gridded
        self.umode_i = umode_i
        self.umodeedges = umodeedges
        self.kperpedges = kperpedges
        self.k1dedges = k1dedges
        self.int_steps = int_steps
        self.verbose = verbose
        self.approxlim = approxlim
        self.mask = mask
        self.sigma_cut = sigma_cut
        self.window = window
        if vis_2 is None:
            self.vis_2 = vis_gridded
        else:
            self.vis_2 = vis_2
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        
    def kperparr(self):
        """The transverse scale of the 2d power spectrum"""
        if self.kperpedges is None:
            kmodeedges = (2*np.pi*self.umodeedges/self.sp.X_0())
        else:
            kmodeedges = self.kmodeedges
        return (kmodeedges[1:]+kmodeedges[:-1])/2
    
    def kparaarr(self):
        """The los scale of the 2d power spectrum"""
        result = (self.sp.eta_arr()*(2*np.pi*self.sp.cosmo.H0
                                     *f_21*units.Hz
                                     *self.sp.cosmo.efunc(self.sp.z_0())
                                     /constants.c/(1+self.sp.z_0())**2)
                  .to("Hz/Mpc").value)
        return result
    
    def delay_power(self):
        """Calculate the delay power spectrum"""
        pdgamma,umodecen,etacen = vis_power(self.sp,self.umode_i,
                                            self.umodeedges,self.vis_gridded,vis_2 = self.vis_2,
                                            device = self.device,
                                            verbose = self.verbose,
                                           sigma_cut = self.sigma_cut,window = self.window)
        if np.sum(np.isnan(pdgamma))>0:
            warnings.warn("pdgamma has nan element. Try coarser uedges.")
        
        return pdgamma,umodecen,etacen
    
    def conv_matrix(self):
        """Calculate the convertion matrix"""
        result = inversemab(self.sp,self.umode_i,self.umodeedges,
                            int_steps = self.int_steps, 
                            approxlim = self.approxlim, 
                            kmodeedges = self.kperpedges, 
                            device = self.device,verbose = self.verbose)
        return result
    
    def tps(self,pdgamma=None,conv_matrix=None,diag=False):
        """Calculate the temperature power spectrum"""
        if self.verbose:
            start_time = time.time()
            print("tps:start calculating the power spectrum")
        if pdgamma is None:
            pdgamma,umodecen,etacen = self.delay_power()
        if conv_matrix is None:
            conv_matrix = self.conv_matrix()
        
        pdgamma = torch.from_numpy(pdgamma).float().to(self.device)
        conv_matrix = torch.from_numpy(conv_matrix).float().to(self.device)
        if diag:
            conv_matrix = torch.diag(torch.diagonal(conv_matrix))
        result = torch.matmul(
            conv_matrix[None,:,:]
            ,pdgamma[:,:,None]).reshape((self.sp.num_channels,-1))
        # if only 2d power is needed
        if self.k1dedges is None:
            pdgamma = 0
            conv_matrix = 0
            tps2d = result.cpu().numpy()
            result = 0 
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            if self.verbose:
                print(
                    "ps calculation finished! Time spent: %f seconds" 
                    % (time.time()-start_time))
            return tps2d
        else:
            mask = torch.from_numpy(np.array(self.mask)).to(self.device)
            mask = torch.flatten(mask)[:,None]
            k1dbin = torch.from_numpy(self.k1dedges).to(self.device)
            kperpcen = torch.from_numpy(self.kperparr()).to(self.device)
            kparacen = torch.from_numpy(self.kparaarr()).to(self.device)
            k1d_i = torch.flatten(torch.sqrt(kperpcen[None,:]**2
                                             +kparacen[:,None]**2))
            weight = ((k1d_i[:,None]>k1dbin[None,:-1])
                      *(k1d_i[:,None]<k1dbin[None,1:]))*mask
            result = torch.flatten(result)
            result1d = (torch.sum(result[:,None]*weight,axis=0)
                        /torch.sum(weight,axis=0))
            #clear memory
            tps2d = result.reshape((self.sp.num_channels,-1)).cpu().numpy()
            result = 0
            pdgamma = 0
            conv_matrix = 0
            k1dbin = 0
            kperpcen = 0
            kparacen = 0
            k1d_i = 0
            weight = 0
            tps1d = result1d.cpu().numpy()
            result1d = 0
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            if self.verbose:
                print(
                    "ps calculation finished! Time spent: %f seconds" 
                    % (time.time()-start_time))
            return tps2d,tps1d