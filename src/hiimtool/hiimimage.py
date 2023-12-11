import numpy as np
from astropy import constants,units
import time
import torch
from torch.fft import fft,fftn,fftfreq
import warnings
from itertools import product
import psutil
from .util import histogramdd
from scipy.linalg import sqrtm

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
    # grid the density field
    den= grid_im(pos,len_side,N,weights=weights,
                   verbose=verbose,norm=norm,)
    
    len_grid = len_side/(N-1)

    if pos2 is None:
        if weights2 is None:
            den2 = den
        else:
            den2= grid_im(pos,len_side,N,weights=weights2,
                   verbose=verbose,norm=norm,)
            flag12=True
    else:
        if weights2 is None:
            den2= grid_im(pos2,len_side,N,weights=weights,
                   verbose=verbose,norm=norm,)
        else:
            den2= grid_im(pos2,len_side,N,weights=weights2,
                   verbose=verbose,norm=norm,)
        flag12 = True
    den = torch.from_numpy(den).to(device)
    den2 = torch.from_numpy(den2).to(device)
    #clear memory
    bins = 0
    pos = 0
    pos2 = 0
    weights = 0
    weights2 = 0
    lenxbin = 0
    lenybin = 0
    lenzbin = 0
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

def imdeconv(image,beam,psf,findx = 0):
    """
    Take out the primary beam term. 
    Deconvolve the psf and reconvolve with the worst psf (default the first channel)
    """
    num_ch = image.shape[0]
    imsize = image.shape[1]
    tdecov = np.zeros_like(image)
    pos = np.zeros((num_ch,imsize,imsize,3))
    tdecov = np.zeros_like(image)
    for i in range(num_ch):
        tfft = np.fft.fftn(image[i]/beam[i]**2)
        psf_f = np.fft.fftn(psf[i])
        tdecov[i] = np.fft.fftshift(np.fft.ifftn(tfft/psf_f)).real
    trecov = np.zeros_like(image)
    psf_f = np.fft.fftn(psf[findx])
    for i in range(num_ch):
        tfft = np.fft.fftn(tdecov[i])
        trecov[i] = np.fft.fftshift(np.fft.ifftn(tfft*psf_f)).real
    return trecov

def grid_im(
    pos,len_side,N,weights=None,
    device=None,norm=False,verbose=False,center=True):
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
    
    len_grid = len_side/(N-1)
    lenxbin = torch.linspace(-len_grid[0]/2,len_side[0]+len_grid[0]/2,
                             steps=N[0]+1,device=device)
    lenybin = torch.linspace(-len_grid[1]/2,len_side[1]+len_grid[1]/2,
                             steps=N[1]+1,device=device)
    lenzbin = torch.linspace(-len_grid[2]/2,len_side[2]+len_grid[2]/2,
                             steps=N[2]+1,device=device)
    pos = torch.from_numpy(pos).to(device)
    weights = torch.from_numpy(weights).to(device)
    den,bins = histogramdd(pos.T,bins=[lenxbin,lenybin,lenzbin],weights=weights)
    #clear memory
    bins = 0.0
    pos = 0
    weights = 0
    lenxbin = 0
    lenybin = 0
    lenzbin = 0
    den = den.float() # reduce size
    if norm:
        den = ((den-den.mean())/den.mean()).float()
    else:
        if center:
            den = (den-den.mean()).float()
    if device != 'cpu':
        torch.cuda.empty_cache()
    den = den.cpu().numpy()
    return den

def get_f_density(
    pos,len_side,N,weights=None,
    device=None,norm=True,verbose=False,window=None,):
    """Calculate the 3d Fourier density given a discrete sample of sources"""
    if verbose:
        start_time = time.time()
        print("get_f_density:start calculating the Fourier density")
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
    # grid the density field
    den= grid_im(pos,len_side,N,weights=weights,
                   verbose=verbose,norm=norm,)
    
    len_grid = len_side/(N-1)
    den = torch.from_numpy(den).to(device)
    #clear memory
    bins = 0
    pos = 0
    weights = 0
    lenxbin = 0
    lenybin = 0
    lenzbin = 0
    vbox = np.product(len_side)
    den = fftn(den*window[None,None,:])
    den = np.product(len_grid)*den/vbox
    den = den*renorm
    kxvec = 2*np.pi*fftfreq(N[0],d=len_grid[0],device=device)
    kyvec = 2*np.pi*fftfreq(N[1],d=len_grid[1],device=device)
    kzvec = 2*np.pi*fftfreq(N[2],d=len_grid[2],device=device)   
    #clear memory
    indx = 0
    kperpedges = 0
    kperpfield = 0
    kxvec = kxvec.cpu().numpy()
    kyvec = kyvec.cpu().numpy()
    kzvec = kzvec.cpu().numpy()
    den = den.cpu().numpy()
    if device != 'cpu':
        torch.cuda.empty_cache()
    if verbose:
        print(
            "density calculation finished! Time spent: %f seconds" % 
            (time.time()-start_time))
    return den,kxvec,kyvec,kzvec

def dft_matrix(num_p,device=None):
    """The 1-D DFT kernel with forward renormalisation"""
    # the ordinary dft kernel for each u-v grid
    if device is None:
        device = 'cpu'
    marr = torch.linspace(0,num_p-1,num_p).to(torch.complex128).to(device)
    dft_block = torch.exp(-2*np.pi*1j*marr[:,None]*marr[None,:]/num_p)
    marr = 0.0
    if device != 'cpu':
        torch.cuda.empty_cache()
    return dft_block/num_p

def C_alpha(num_ch,device=None):
    if device is None:
        device = 'cpu'
    idenm = torch.diag(
            torch.ones(num_ch)).type(torch.complex128).to(device)
    diagwa = torch.einsum('ai,ij->aij',idenm,idenm)
    result = dft_matrix(num_ch,device=device)
    result = torch.einsum('ij,ajk,kl->ail',result,diagwa,
                         torch.conj(result.t()))*num_ch
    diagwa = 0
    idenm = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    return result

def H_alpha_beta(num_ch,R_mat=None,device=None):
    if device is None:
        device = 'cpu'
    if R_mat is None:
        R_mat = torch.diag(torch.ones(num_ch)).type(torch.complex128).to(device)
    calpha=C_alpha(num_ch,device=device)
    Hab = torch.einsum('ij,ajk,kl,bli->ab',
                       torch.conj(R_mat.t()),
                       calpha,
                       R_mat,
                       calpha
                      )
    return Hab

def estimator(num_ch,R_mat=None,device=None,renorm=''):
    if device is None:
        device = 'cpu'
    e_mat = dft_matrix(num_ch,device=device)
    if R_mat is None:
        R_mat = torch.diag(torch.ones(num_ch)).type(torch.complex128).to(device)
    idenm = torch.diag(
            torch.ones(num_ch)).type(torch.complex128).to(device)
    diagwa = torch.einsum('ai,ij->aij',idenm,idenm)
    e_mat = torch.einsum(
        'ij,jk,akl,lm,mn->ain',
        torch.conj(R_mat).t(),
        torch.conj(e_mat).t(),
        diagwa,
        e_mat,
        R_mat
    )
    Hab = H_alpha_beta(num_ch,R_mat=R_mat,device=device)
    if renorm == 'sqrtm':
        mab = sqrtm(Hab.cpu().numpy())
        mab = torch.from_numpy(mab).to(device)
    else:
        mab = torch.diag(1/Hab.sum(dim=0))
    e_mat = torch.einsum('ab,bij->aij',mab,e_mat)
    wab = mab@Hab
    factor = 1/(wab.sum()/num_ch)
    wab = wab*factor
    e_mat = e_mat*factor
    mab = 0
    Hab = 0
    idenm = 0
    diagwa = 0
    R_mat = 0
    if device != 'cpu':
        torch.cuda.empty_cache()
    return e_mat,wab