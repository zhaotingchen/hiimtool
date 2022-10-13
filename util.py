import numpy as np
from astropy import constants,units
import time
import warnings
from itertools import product
import psutil
import torch
from torch.fft import fft,fftn,fftfreq
from hiimvis import Specs

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

def calcov(im,weights=None,im2=None,weights2=None):
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
    
    cov = (torch.einsum('ia,ia,ja,ja->ij',im,weights,torch.conj(im2),weights2)/
           torch.einsum('ia,ja->ij',weights,weights2))
    return cov

