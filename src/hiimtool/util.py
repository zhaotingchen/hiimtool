import numpy as np
from astropy import constants,units
import time
import warnings
from itertools import product
import torch
from torch.fft import fft,fftn,fftfreq
#from hiimvis import Specs

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

def slicer_vectorized(a,start,end):
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.frombuffer(b.tobytes(),dtype=(str,end-start))

def calcov(im,weights=None,im2=None,weights2=None,split=None):
    '''calculate the covariance of a data set, assuming the first axis is the frequency'''
    num_ch = im.shape[0]
    im = im.reshape((num_ch,-1))
    if im2 is None:
        im2 = im
    im2 = im2.reshape((num_ch,-1))
    num_pix = im.shape[1]
    im = torch.from_numpy(im).to(torch.complex64)
    im2 = torch.from_numpy(im2).to(torch.complex64)
    if weights is None:
        weights = torch.ones_like(im).to(torch.complex64)
    else:
        weights = torch.from_numpy(weights).to(torch.complex64)
    if weights2 is None:
        weights2 = weights
    else:
        weights2 = torch.from_numpy(weights2).to(torch.complex64)
    weights = weights.reshape((num_ch,-1))
    
    if split is None:
        cov = (torch.einsum('ia,ia,ja,ja->ij',im,weights,torch.conj(im2),weights2)/
           torch.einsum('ia,ja->ij',weights,weights2))
        cov = cov.cpu().numpy()
    else:
        split = int(split)
        cov = torch.zeros((num_ch,num_ch)).to(torch.complex64)
        weight_sum = torch.zeros((num_ch,num_ch)).to(torch.complex64)
        im = torch.split(im,split,dim=-1)
        im2 = torch.split(im2,split,dim=-1)
        weights = torch.split(weights,split,dim=-1)
        weights2 = torch.split(weights2,split,dim=-1)
        for i in range(len(im)):
            cov+=torch.einsum('ia,ia,ja,ja->ij',im[i],weights[i],torch.conj(im2[i]),weights2[i])
            weight_sum += torch.einsum('ia,ja->ij',weights[i],weights2[i])
        cov = cov/weight_sum
        weight_sum = 0.0
        cov = cov.cpu().numpy()
    return cov




_range = range

def histogramdd(sample,bins=None,range=None,weights=None,remove_overflow=True):
    """
    This is directly taken from https://github.com/miranov25/RootInteractive/blob/b54446e09072e90e17f3da72d5244a20c8fdd209/RootInteractive/Tools/Histograms/histogramdd.py
    """
    edges=None
    device=None
    custom_edges = False
    D,N = sample.shape
    if device == None:
        device = sample.device
    if bins == None:
        if edges == None:
            bins = 10
            custom_edges = False
        else:
            try:
                bins = edges.size(1)-1
            except AttributeError:
                bins = torch.empty(D)
                for i in _range(len(edges)):
                    bins[i] = edges[i].size(0)-1
                bins = bins.to(device)
            custom_edges = True
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except AttributeError:
        # bins is either an integer or a list
        if type(bins) == int:
            bins = torch.full([D],bins,dtype=torch.long,device=device)
        elif torch.is_tensor(bins[0]):
            custom_edges = True
            edges = bins
            bins = torch.empty(D,dtype=torch.long)
            for i in _range(len(edges)):
                bins[i] = edges[i].size(0)-1
            bins = bins.to(device)
        else:
            bins = torch.as_tensor(bins)
    if bins.dim() == 2:
        custom_edges = True
        edges = bins
        bins = torch.full([D],bins.size(1)-1,dtype=torch.long,device=device)
    if custom_edges:
        use_old_edges = False
        if not torch.is_tensor(edges):
            use_old_edges = True
            edges_old = edges
            m = max(i.size(0) for i in edges)
            tmp = torch.empty([D,m],device=edges[0].device)
            for i in _range(D):
                s = edges[i].size(0)
                tmp[i,:]=edges[i][-1]
                tmp[i,:s]=edges[i][:]
            edges = tmp.to(device)
        k = torch.searchsorted(edges,sample)
        k = torch.min(k,(bins+1).reshape(-1,1))
        if use_old_edges:
            edges = edges_old
        else:
            edges = torch.unbind(edges)
    else:
            if range == None: #range is not defined
                range = torch.empty(2,D,device=device)
                if N == 0: #Empty histogram
                    range[0,:] = 0
                    range[1,:] = 11-Copy1
                else:
                    range[0,:]=torch.min(sample,1)[0]
                    range[1,:]=torch.max(sample,1)[0]
            elif not torch.is_tensor(range): #range is a tuple
                r = torch.empty(2,D)
                for i in _range(D):
                    if range[i] is not None:
                        r[:,i] = torch.as_tensor(range[i])
                    else:
                        if N == 0: #Edge case: empty histogram
                            r[0,i] = 0
                            r[1,i] = 1
                        r[0,i]=torch.min(sample[:,i])[0]
                        r[1,i]=torch.max(sample[:,i])[0]
                range = r.to(device=device,dtype=sample.dtype)
            singular_range = torch.eq(range[0],range[1]) #If the range consists of only one point, pad it upchmod +x eduroam-linux-TUoM.py 
            range[0,singular_range] -= .5
            range[1,singular_range] += .5
            edges = [torch.linspace(range[0,i],range[1,i],bins[i]+1) for i in _range(len(bins))]
            tranges = torch.empty_like(range)
            tranges[1,:] = bins/(range[1,:]-range[0,:])
            tranges[0,:] = 1-range[0,:]*tranges[1,:]
            k = torch.addcmul(tranges[0,:].reshape(-1,1),sample,tranges[1,:].reshape(-1,1)).long() #Get the right index
            k = torch.max(k,torch.zeros([],device=device,dtype=torch.long)) #Underflow bin
            k = torch.min(k,(bins+1).reshape(-1,1))


    multiindex = torch.ones_like(bins)
    multiindex[1:] = torch.cumprod(torch.flip(bins[1:],[0])+2,-1).long()
    multiindex = torch.flip(multiindex,[0])
    l = torch.sum(k*multiindex.reshape(-1,1),0)
    hist = torch.bincount(l,minlength=(multiindex[0]*(bins[0]+2)).item(),weights=weights)
    hist = hist.reshape(tuple(bins+2))
    if remove_overflow:
        core = D * (slice(1, -1),)
        hist = hist[core]
    return hist,edges

def bin_3d_to_1d(ps3d,kfield,k1dedges,device='cpu',weights=None,error=False):
    """Bin a 3-D power spectrum into 1-D"""
    ps3d = np.ravel(ps3d)
    kfield = np.ravel(kfield)
    ps3d = torch.from_numpy(ps3d).to(device)
    if weights is None:
        weights = torch.ones_like(ps3d)
    else:
        weights = np.ravel(weights)
        weights = torch.from_numpy(weights).to(device)
    kfield = torch.from_numpy(kfield).to(device)
    k1dcen = (k1dedges[1:]+k1dedges[:-1])/2
    k1dedges = torch.from_numpy(k1dedges).to(device)
    indx = (kfield[:,None]>=k1dedges[None,:-1])*(kfield[:,None]<k1dedges[None,1:])
    ps1d = torch.sum(ps3d[:,None]*indx*weights[:,None],dim=(0))/torch.sum(indx*weights[:,None],dim=(0))
    if error is True:
        ps1derr = torch.sqrt(torch.sum((ps3d[:,None]-ps1d[None,:])**2*(indx*weights[:,None])**2,dim=(0))/torch.sum((indx*weights[:,None]),dim=(0))**2)
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