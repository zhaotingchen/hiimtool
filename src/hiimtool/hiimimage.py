import numpy as np
import healpy as hp
from astropy import constants,units
import time
import torch
from torch.fft import fft,fftn,fftfreq
import warnings
import progressbar
from itertools import product
import psutil
from .util import histogramdd

def get_power_cy(
    pos,kperpedges,len_side,N,weights=None,
    pos2=None,weights2=None,error=False,
    device=None,norm=True,verbose=False,window=None,):
    """Calculate the 1d power spectrum given a discrete sample of sources"""
    if verbose:
        start_time = time.time()
        print("get_power:start calculating the power spectrum")
    num_sources,num_dim = pos.shape
    if (len(len_side) != num_dim):
        raise ValueError('The dimensions of samples and the dimensions' 
                         'of the grids does not match')
    if device is None:
        device = 'cpu'
        
    if weights is None:
        weights = np.ones(pos.shape[0])
    
    N = np.array([N]).reshape((-1)).astype('int')
    if len(N) ==1:
        N = np.ones(3).astype('int')*N
    
    if window is None:
        window = np.ones(N[-1])
        renorm = 1.0
    else:
        testarr_f = np.zeros(N[-1])
        testarr_f[N[-1]//2]=1.0
        testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
        testarr_w = (np.fft.fft(testarr*window))
        renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    window = torch.from_numpy(window).to(device)
    
    len_grid = len_side/(N-1)
    lenxbin = torch.linspace(-len_grid[0]/2,len_side[0]+len_grid[0]/2,
                             steps=N[0]+1,device=device)
    lenybin = torch.linspace(-len_grid[1]/2,len_side[1]+len_grid[1]/2,
                             steps=N[1]+1,device=device)
    lenzbin = torch.linspace(-len_grid[2]/2,len_side[2]+len_grid[2]/2,
                             steps=N[2]+1,device=device)
    pos = torch.from_numpy(pos).to(device)
    if pos2 is None:
        pos2 = pos
    else:
        pos2 = torch.from_numpy(pos2).to(device)
    weights = torch.from_numpy(weights).to(device)
    if weights2 is not None:
        weights2 = torch.from_numpy(weights2).to(device)
    den,bins = histogramdd(pos.T,bins=[lenxbin,lenybin,lenzbin],weights=weights)
    if weights2 is None:
        den2,bins = histogramdd(pos2.T,bins=[lenxbin,lenybin,lenzbin],weights=weights)
    else:
        den2,bins = histogramdd(pos2.T,bins=[lenxbin,lenybin,lenzbin],weights=weights2)
    #clear memory
    bins = 0
    pos = 0
    pos2 = 0
    weights = 0
    weights2 = 0
    lenxbin = 0
    lenybin = 0
    lenzbin = 0
    den = den.float() # reduce size
    #density to delta
    if norm:
        den = ((den-den.mean())/den.mean()).float()
        den2 = ((den2-den2.mean())/den2.mean()).float()
    else:
        den = (den-den.mean()).float()
        den2 = (den2-den2.mean()).float()
    vbox = np.product(len_side)
    den = fftn(den*window[None,None,:])
    den2 = fftn(den2*window[None,None,:])
    den = np.product(len_grid)*den/vbox
    den2 = np.product(len_grid)*den2/vbox
    den = (den*torch.conj(den2)).real.float()*renorm
    if verbose:
        print("get_power: 3D ps calculation finished! To 1D:")
    kxvec = 2*np.pi*fftfreq(N[0],d=len_grid[0],device=device)
    kyvec = 2*np.pi*fftfreq(N[1],d=len_grid[1],device=device)
    kzvec = 2*np.pi*fftfreq(N[2],d=len_grid[2],device=device)   
    kperpfield = torch.sqrt(kxvec[:,None]**2+kyvec[None,:]**2)
    kperpedges = torch.from_numpy(kperpedges).to(device)
    if verbose:
        irange = range((len(kperpedges)-1))
    else:
        irange = range((len(kperpedges)-1))
    p1d = np.zeros((len(kperpedges)-1,len(kzvec)))
    if error:
        p1d_err = np.zeros((len(kperpedges)-1,len(kzvec)))
    for i in irange:
        indx = (kperpfield>=kperpedges[i]) * (kperpfield<kperpedges[i+1])
        p1d[i] = (torch.mean(den[indx],dim=0)*vbox).cpu().numpy()
        if error:
            #p1d_err[i] = (torch.std(den[indx])*vbox).cpu().numpy()
            p1d_err[i] = (torch.mean(den[indx],dim=0)*vbox).cpu().numpy()/np.sqrt(len(den[indx]))
            #print(len(den[indx]))
    kcen = ((kperpedges[1:]+kperpedges[:-1])/2).cpu().numpy()
    #clear memory
    indx = 0
    kperpedges = 0
    kperpfield = 0
    kxvec = 0
    kyvec = 0
    kzvec = kzvec.cpu().numpy()
    den = 0
    den2 = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    if verbose:
        print(
            "ps calculation finished! Time spent: %f seconds" % 
            (time.time()-start_time))
    if error:
        return p1d,p1d_err,kcen,kzvec
    else:
        return p1d,kcen,kzvec

def get_power_1d(
    pos,kedges,len_side,N,weights=None,
    pos2=None,weights2=None,error=False,
    device=None,norm=True,verbose=False,window=None,):
    """Calculate the 1d power spectrum given a discrete sample of sources"""
    if verbose:
        start_time = time.time()
        print("get_power:start calculating the power spectrum")
    num_sources,num_dim = pos.shape
    if (len(len_side) != num_dim):
        raise ValueError('The dimensions of samples and the dimensions' 
                         'of the grids does not match')
    if device is None:
        device = 'cpu'
        
    if weights is None:
        weights = np.ones(pos.shape[0])
    
    N = np.array([N]).reshape((-1)).astype('int')
    if len(N) ==1:
        N = np.ones(3).astype('int')*N
    if window is None:
        window = np.ones(N[-1])
        renorm = 1.0
    else:
        testarr_f = np.zeros(N[-1])
        testarr_f[N[-1]//2]=1.0
        testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
        testarr_w = (np.fft.fft(testarr*window))
        renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    window = torch.from_numpy(window).to(device)
    
    len_grid = len_side/(N-1)
    lenxbin = torch.linspace(-len_grid[0]/2,len_side[0]+len_grid[0]/2,
                             steps=N[0]+1,device=device)
    lenybin = torch.linspace(-len_grid[1]/2,len_side[1]+len_grid[1]/2,
                             steps=N[1]+1,device=device)
    lenzbin = torch.linspace(-len_grid[2]/2,len_side[2]+len_grid[2]/2,
                             steps=N[2]+1,device=device)
    pos = torch.from_numpy(pos).to(device)
    if pos2 is None:
        pos2 = pos
    else:
        pos2 = torch.from_numpy(pos2).to(device)
    weights = torch.from_numpy(weights).to(device)
    if weights2 is not None:
        weights2 = torch.from_numpy(weights2).to(device)
    den,bins = histogramdd(pos.T,bins=[lenxbin,lenybin,lenzbin],weights=weights)
    if weights2 is None:
        den2,bins = histogramdd(pos2.T,bins=[lenxbin,lenybin,lenzbin],weights=weights)
    else:
        den2,bins = histogramdd(pos2.T,bins=[lenxbin,lenybin,lenzbin],weights=weights2)
    #clear memory
    bins = 0
    pos = 0
    pos2 = 0
    weights = 0
    weights2 = 0
    lenxbin = 0
    lenybin = 0
    lenzbin = 0
    den = den.float() # reduce size
    #density to delta
    if norm:
        den = ((den-den.mean())/den.mean()).float()
        den2 = ((den2-den2.mean())/den2.mean()).float()
    else:
        den = (den-den.mean()).float()
        den2 = (den2-den2.mean()).float()
    vbox = np.product(len_side)
    den = fftn(den*window[None,None,:])
    den2 = fftn(den2*window[None,None,:])
    den = np.product(len_grid)*den/vbox
    den2 = np.product(len_grid)*den2/vbox
    den = (den*torch.conj(den2)).real.float()*renorm
    if verbose:
        print("get_power: 3D ps calculation finished! To 1D:")
    kxvec = 2*np.pi*fftfreq(N[0],d=len_grid[0],device=device)
    kyvec = 2*np.pi*fftfreq(N[1],d=len_grid[1],device=device)
    kzvec = 2*np.pi*fftfreq(N[2],d=len_grid[2],device=device)   
    kfield = torch.sqrt(kxvec[:,None,None]**2
                        +kyvec[None,:,None]**2+kzvec[None,None,:]**2)
    kedges = torch.from_numpy(kedges).to(device)
    if verbose:
        irange = range((len(kedges)-1))
    else:
        irange = range((len(kedges)-1))
    p1d = np.zeros((len(kedges)-1))
    if error:
        p1d_err = np.zeros((len(kedges)-1))
    for i in irange:
        indx = torch.where((kfield>=kedges[i]) & (kfield<kedges[i+1]))
        p1d[i] = (torch.mean(den[indx])*vbox).cpu().numpy()
        if error:
            #p1d_err[i] = (torch.std(den[indx])*vbox).cpu().numpy()
            p1d_err[i] = (torch.mean(den[indx])*vbox).cpu().numpy()/np.sqrt(len(den[indx]))
            print(len(den[indx]))
    kcen = ((kedges[1:]+kedges[:-1])/2).cpu().numpy()
    #clear memory
    indx = 0
    kedges = 0
    kfield = 0
    kxvec = 0
    kyvec = 0
    kzvec = 0
    den = 0
    den2 = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    if verbose:
        print(
            "ps calculation finished! Time spent: %f seconds" % 
            (time.time()-start_time))
    if error:
        return p1d,p1d_err,kcen
    else:
        return p1d,kcen
    
def get_power_3d(
    pos,len_side,N,weights=None,
    pos2=None,weights2=None,k_weights1 = None,k_weights2 = None,
    device=None,norm=True,verbose=False,window=None,):
    """Calculate the 3d power spectrum given a discrete sample of sources"""
    flag12 = False
    if verbose:
        start_time = time.time()
        print("get_power:start calculating the power spectrum")
    num_sources,num_dim = pos.shape
    if (len(len_side) != num_dim):
        raise ValueError('The dimensions of samples and the dimensions' 
                         'of the grids does not match')
    if device is None:
        device = 'cpu'
        
    if weights is None:
        weights = np.ones(pos.shape[0])
    
    N = np.array([N]).reshape((-1)).astype('int')
    if len(N) ==1:
        N = np.ones(3).astype('int')*N
    
    if window is None:
        window = np.ones(N[-1])
        renorm = 1.0
    else:
        testarr_f = np.zeros(N[-1])
        testarr_f[N[-1]//2]=1.0
        testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
        testarr_w = (np.fft.fft(testarr*window))
        renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
    window = torch.from_numpy(window).to(device)
    len_grid = len_side/(N-1)
    lenxbin = torch.linspace(-len_grid[0]/2,len_side[0]+len_grid[0]/2,
                             steps=N[0]+1,device=device)
    lenybin = torch.linspace(-len_grid[1]/2,len_side[1]+len_grid[1]/2,
                             steps=N[1]+1,device=device)
    lenzbin = torch.linspace(-len_grid[2]/2,len_side[2]+len_grid[2]/2,
                             steps=N[2]+1,device=device)
    pos = torch.from_numpy(pos).to(device)
    if pos2 is None:
        pos2 = pos
    else:
        pos2 = torch.from_numpy(pos2).to(device)
        flag12 = True
    weights = torch.from_numpy(weights).to(device)
    den,bins = histogramdd(pos.T,bins=[lenxbin,lenybin,lenzbin],weights=weights)
    if weights2 is None:
        den2,bins = histogramdd(pos2.T,bins=[lenxbin,lenybin,lenzbin],weights=weights)
    else:
        flag12 = True
        weights2 = torch.from_numpy(weights2).to(device)
        den2,bins = histogramdd(pos2.T,bins=[lenxbin,lenybin,lenzbin],weights=weights2)
    #clear memory
    bins = 0
    pos = 0
    pos2 = 0
    weights = 0
    weights2 = 0
    lenxbin = 0
    lenybin = 0
    lenzbin = 0
    #den = den.float() # reduce size
    #density to delta
    if norm:
        den = ((den-den.mean())/den.mean()).float()
        den2 = ((den2-den2.mean())/den2.mean()).float()
    else:
        den = (den-den.mean()).float()
        den2 = (den2-den2.mean()).float()
    vbox = np.product(len_side)
    den = fftn(den*window[None,None,:])
    den2 = fftn(den2*window[None,None,:])
    den = np.product(len_grid)*den/vbox
    den2 = np.product(len_grid)*den2/vbox
    # if there is k-weights (e.g. psf)
    if k_weights1 is not None:
        k_weights1 = torch.from_numpy(k_weights1).to(device)
    else:
        k_weights1 = torch.ones_like(den)
    den = den*k_weights1
    if k_weights2 is not None:
        k_weights2 = torch.from_numpy(k_weights2).to(device)
    else:
        if flag12: # if cross-power, psf can be different
            k_weights2 = torch.ones_like(den2)
        else: # if auto_power, same psf
            k_weights2 = k_weights1
    den2 = den2*k_weights2
    den = (den*torch.conj(den2)).real.float()*renorm*vbox
    if verbose:
        print("get_power: 3D ps calculation finished! To 1D:")
    kxvec = 2*np.pi*fftfreq(N[0],d=len_grid[0],device=device)
    kyvec = 2*np.pi*fftfreq(N[1],d=len_grid[1],device=device)
    kzvec = 2*np.pi*fftfreq(N[2],d=len_grid[2],device=device)   
    #clear memory
    indx = 0
    kperpedges = 0
    kperpfield = 0
    k_weights1 = 0
    k_weights2 = 0
    kxvec = kxvec.cpu().numpy()
    kyvec = kyvec.cpu().numpy()
    kzvec = kzvec.cpu().numpy()
    den = den.cpu().numpy()
    den2 = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    if verbose:
        print(
            "ps calculation finished! Time spent: %f seconds" % 
            (time.time()-start_time))
    return den,kxvec,kyvec,kzvec