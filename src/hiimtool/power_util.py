'''
Modules for computation of the power spectrum in images/simulations.
'''
import numpy as np
from .hiimimage import grid_im
from torch.fft import fft,fftn,fftfreq
from functools import cached_property
import torch

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


class MultiPower():
    def __init__(self, 
                 pos,
                 len_side,
                 N_side,
                 cosmo,
                 field=None,
                 kperpedges=None,
                 k1dedges=None,
                 verbose=False,
                 window=None,
                 los_axis=-1,
                 norm=False,
                 device=None,
                 los_0=None,
                 density=False,
                ):
        self.pos = pos.reshape((-1,3))
        self.len_side = len_side
        N_side = np.array([N_side]).reshape((-1)).astype('int')
        self.N_side = N_side
        self.los_axis = los_axis
        if window is None:
            window = np.ones(N_side[los_axis])
        if len(window)!=N_side[los_axis]:
            raise ValueError('length of the window does not match the los dimension')
        self.window = window
        len_grid = len_side/(N_side)
        self.len_grid = len_grid
        self.cosmo=cosmo
        self.kxvec = 2*np.pi*np.fft.fftfreq(N_side[0],d=len_grid[0])
        self.kyvec = 2*np.pi*np.fft.fftfreq(N_side[1],d=len_grid[1])
        self.kzvec = 2*np.pi*np.fft.fftfreq(N_side[2],d=len_grid[2])   
        self.vbox = np.prod(len_side)
        self.vgrid = np.prod(len_grid)
        self.field = field
        if self.field is not None:
            self.field = np.ravel(self.field)
        self.kperpedges = kperpedges
        self.k1dedges = k1dedges
        self.verbose = verbose
        if los_axis < 0:
            los_axis = 3+los_axis
            
        self.los_axis = los_axis
        self.norm=norm
        self.density=density
        if device is None:
            device = 'cpu'
        self.device = device
        self.los_0 = los_0

    def grid_field(self,field):
        den= grid_im(self.pos,self.len_side,self.N_side,weights=field,density=self.density,
                   verbose=self.verbose,norm=self.norm,device=self.device)
        return den

    @cached_property
    def den_grid(self):
        return self.grid_field(self.field)
    
    def f_density(self,field):
        N = self.N_side
        los = self.los_axis
        vbox = self.vbox
        window = self.window
        device = self.device
        axis = [0,1,2]
        # make sure los is the last axis
        axis.remove(los)
        axis = axis+[los,]
        #print(axis)
        # calculate a naive renormalization
        testarr_f = np.zeros(N[los])
        testarr_f[N[los]//2]=1.0
        testarr = np.fft.fftshift(np.fft.ifft(testarr_f))
        testarr_w = (np.fft.fft(testarr*window))
        renorm = (np.abs(testarr_f)**2).sum()/(np.abs(testarr_w)**2).sum()
        den = np.transpose(field,axes=axis)
        den = torch.from_numpy(den).to(device)
        window = torch.from_numpy(window).to(device)
        den = fftn(den*window[None,None,:])*np.sqrt(renorm)
        # transpose back
        den = np.prod(self.len_grid)*den/self.vbox
        den = den.cpu().numpy()
        den = np.transpose(den,axes=axis)
        window = 0.0
        if device != 'cpu':
            torch.cuda.empty_cache()
        return den

    @cached_property
    def delta_f_0(self):
        return self.f_density(self.den_grid)

    @cached_property
    def power_3d_0(self):
        delta_f_0 = self.delta_f_0
        delta_f_0 = torch.from_numpy(delta_f_0).to(self.device)
        power = torch.abs(delta_f_0)**2*self.vbox
        power = power.cpu().numpy()
        delta_f_0 = delta_f_0.cpu().numpy()
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        return power

    @cached_property
    def delta_f_2(self):
        if self.los_0 is None:
            xz_cos = 1
        else:
            x_pos = self.pos.copy()
            x_pos[:,self.los_axis] += self.los_0
            x_mode = np.sqrt((x_mode**2).sum(axis=-1))
            xz_cos = x_pos[:,self.los_axis]/x_mode
        axis = [0,1,2]
        # make sure los is the last axis
        los = self.los_axis
        axis.remove(los)
        axis = axis+[los,]
        den_2 = self.grid_field(xz_cos**2*self.field)
        Q_zz = self.f_density(den_2)
        Q_zz = np.transpose(Q_zz,axes=axis)
        k_mode = self.k_mode.copy()
        k_mode = np.transpose(k_mode,axes=axis)
        delta_f_0 = self.delta_f_0.copy()
        delta_f_0 = np.transpose(delta_f_0,axes=axis)
        delta_f_2 = np.nan_to_num(3/2*(self.k_para[None,None,:]/self.k_mode)**2*Q_zz-1/2*delta_f_0)
        delta_f_2 = np.transpose(delta_f_2,axes=axis)
        return delta_f_2

    @cached_property
    def power_3d_2(self):
        delta_f_0 = torch.from_numpy(self.delta_f_0).to(self.device)
        delta_f_2 = torch.from_numpy(self.delta_f_2).to(self.device)
        power = torch.real(delta_f_0*torch.conj(delta_f_2))*self.vbox*5
        power = power.cpu().numpy()
        delta_f_0 = delta_f_0.cpu().numpy()
        delta_f_2 = delta_f_2.cpu().numpy()
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        return power

    @cached_property
    def delta_f_4(self):
        if self.los_0 is None:
            xz_cos = 1
        else:
            x_pos = self.pos.copy()
            x_pos[:,self.los_axis] += self.los_0
            x_mode = np.sqrt((x_mode**2).sum(axis=-1))
            xz_cos = x_pos[:,self.los_axis]/x_mode
        axis = [0,1,2]
        # make sure los is the last axis
        los = self.los_axis
        axis.remove(los)
        axis = axis+[los,]
        den_4 = self.grid_field(xz_cos**4*self.field)
        Q_zzzz = self.f_density(den_4)
        Q_zzzz = np.transpose(Q_zzzz,axes=axis)
        k_mode = self.k_mode.copy()
        k_mode = np.transpose(k_mode,axes=axis)
        delta_f_0 = self.delta_f_0.copy()
        delta_f_0 = np.transpose(delta_f_0,axes=axis)
        delta_f_2 = self.delta_f_2.copy()
        delta_f_2 = np.transpose(delta_f_2,axes=axis)
        delta_f_4 = np.nan_to_num(
            35/8*(self.k_para[None,None,:]/self.k_mode)**4*Q_zzzz
            -5/2*delta_f_2-7/8*delta_f_0
        )
        delta_f_4 = np.transpose(delta_f_4,axes=axis)
        return delta_f_4
        
    @cached_property
    def power_3d_4(self):
        delta_f_0 = torch.from_numpy(self.delta_f_0).to(self.device)
        delta_f_4 = torch.from_numpy(self.delta_f_4).to(self.device)
        power = torch.real(delta_f_0*torch.conj(delta_f_4))*self.vbox*9
        power = power.cpu().numpy()
        delta_f_0 = delta_f_0.cpu().numpy()
        delta_f_4 = delta_f_4.cpu().numpy()
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        return power
            
    
    @cached_property
    def k_mode(self):
        k = np.sqrt(self.kxvec[:,None,None]**2+self.kyvec[None,:,None]**2+self.kzvec[None,None,:]**2)
        return k
        
    @cached_property
    def k_perp(self):
        k_perp_arr = [self.kxvec,self.kyvec,self.kzvec,]
        k_perp_arr.pop(self.los_axis)
        k_perp_mode = np.sqrt(k_perp_arr[0][:,None]**2+k_perp_arr[1][None,:]**2)
        return k_perp_mode
    
    @cached_property
    def k_para(self):
        return [self.kxvec,self.kyvec,self.kzvec,][self.los_axis]
    
    def ps_cy(self,ps_3d):
        axis = [0,1,2]
        # make sure los is the first axis
        los = self.los_axis
        axis.remove(los)
        axis = [los,]+axis
        ps_3d = np.transpose(ps_3d,axes=axis)
        ps_cy = bin_3d_to_cy_lowmem(ps_3d.reshape((len(ps_3d),-1)),self.k_perp.reshape(-1),self.kperpedges)
        return ps_cy
        
    @cached_property
    def power_cy_0(self):
        return self.ps_cy(self.power_3d_0)

    @cached_property
    def power_cy_2(self):
        return self.ps_cy(self.power_3d_2)
        
    @cached_property
    def power_cy_4(self):
        return self.ps_cy(self.power_3d_4)