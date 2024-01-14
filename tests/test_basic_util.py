from hiimtool.basic_util import p2dim,chisq,vfind_scan,vfind_id,Specs,fill_nan,f_21,itr_tnsq_avg,delay_transform,get_conv_mat,himf,cal_himf,cumu_nhi_from_himf,sample_from_dist,busy_function_simple,busy_function_0,dft_mat,unravel_list,get_taper_renorm,cal_cov_simple
import pytest
import numpy as np
from astropy.cosmology import Planck15,Planck18
from scipy.signal import blackmanharris
from scipy.special import erf


lamb_21 = 0.21106114054160 # in meters

def func(x,pars):
    a,b,c=pars
    return a*x**2+b*x+c

def uniform_pdf(x):
    return np.ones_like(x)

def uniform_cdf(x):
    return x

def test_p2dim():
    '''
    Test if the `p2dim` function finds the correct Fourier mode.
    '''
    l_start = np.random.choice(np.arange(100))
    for plen in (l_start,l_start+1):
        p2d = np.zeros((1,plen))
        p2d[0] = np.fft.fftfreq(plen)
        result = p2dim(p2d)
        if plen%2==1:
            assert (result!=0).sum()==0
        else:
            assert (result[0,:-1]!=0).sum()==0
            assert result[0,-1]==np.fft.fftfreq(plen).min()
            
def test_find_id():
    '''test `find_id` function'''
    file_arr = np.random.randint(0,9999999999,10)
    file_arr = file_arr.astype('str')
    file_arr = np.char.zfill(file_arr, 10)
    assert (vfind_id(file_arr)!=file_arr).sum()==0
    file_arr = np.random.randint(0,9999999999,2)
    file_arr = file_arr.astype('str')
    file_arr = np.char.zfill(file_arr, 10)
    file_test = file_arr[0]+'/'+file_arr[1]
    with pytest.raises(ValueError):
        vfind_id(file_test)
        
def test_find_scan():
    '''test `find_scan` function'''
    scan_arr = np.random.randint(0,9999,10)
    file_arr = scan_arr.astype('str')
    file_arr = np.char.zfill(file_arr, 4).astype('object')
    for i in range(len(file_arr)):
        file_arr[i] = file_arr[i]+'.'
        file_arr[i] = '.'+file_arr[i]
    assert (vfind_scan(file_arr).astype('int')!=scan_arr).sum()==0
    scan_arr = np.random.randint(0,9999,2)
    file_arr = scan_arr.astype('str')
    file_arr = np.char.zfill(file_arr, 4).astype('object')
    file_test = '.'+file_arr[0]+'./.'+file_arr[1]+'.'
    with pytest.raises(ValueError):
        vfind_scan(file_test)
        
def test_chisq():
    result = chisq((0,0,1),func,np.ones(10),np.ones(10),1)
    assert result == 0
    result = chisq((0,0,1),func,np.ones(10),np.zeros(10),1)
    assert result == 5
    
def test_fill_nan():
    xarr = np.ones(10)
    for i in range(len(xarr)):
        yarr = xarr.copy()
        yarr[i] = np.nan
        assert np.allclose(fill_nan(yarr),xarr)
        
def test_Specs():
    freq_arr = np.linspace(200,250,51)*1e6
    sp = Specs(cosmo=Planck15,freq_start_hz=freq_arr[0],num_channels=len(freq_arr),deltav_ch = np.diff(freq_arr).mean(),FWHM_ref = 1,FWHM_freq_ref = 1)
    assert np.allclose(sp.eta_arr(),np.fft.fftfreq(len(freq_arr),d=np.diff(freq_arr)[0]))
    assert np.allclose(sp.lamb_arr(),lamb_21*(1+sp.z_arr()))
    assert np.allclose(sp.chi_arr(),Planck15.comoving_distance(sp.z_arr()).value)
    assert np.allclose(sp.Hz_arr(),Planck15.H(sp.z_arr()).value)
    assert sp.z_0() == (f_21/freq_arr[0]+f_21/freq_arr[-1]-2)/2
    assert np.allclose(sp.freq_0(),2/(1/freq_arr[0]+1/freq_arr[-1]))
    assert sp.X_0() == Planck15.comoving_distance(sp.z_0()).value
    assert sp.k_para().min() == -0.21842205206818646
    assert sp.k_para().max() == 0.21842205206818646
    assert sp.k_perp(0) == 0
    assert np.allclose(sp.lambda_0(),lamb_21*(1+sp.z_0()))
    
def test_itr_tnsq_avg():
    rand_arr = np.random.normal(0,1,size=(1000,100))
    rand_arr = (rand_arr**2).mean(axis=-1)
    rand_avg = itr_tnsq_avg(rand_arr,10)
    assert np.abs(rand_avg-1)<1e-1
    
    
def test_delay_transform():
    test_arr = np.ones(np.random.randint(1,100,size=1)[0])
    test_f_arr = delay_transform(test_arr,1,test_arr)
    assert test_f_arr[0] == len(test_arr)
    assert np.allclose(test_f_arr[1:],0)
    with pytest.raises(AssertionError) as exc_info:
        delay_transform(test_arr,1,test_arr[:-1])
        
def test_get_conv_mat():
    num_grid = np.random.randint(1,10000)
    test_x = np.arange(num_grid)
    test_w = blackmanharris(num_grid)
    conv_arr = np.convolve(test_w,test_x,mode='same')
    test_conv = get_conv_mat(test_w)@test_x
    assert np.allclose(conv_arr,test_conv)
    
def test_himf():
    h_70 = Planck18.h/0.7
    mmin=6
    phi_star = 4.5*1e-3*h_70**3 # in Mpc-3 dex-1
    m_star = np.log10(10**(9.94)/h_70**2)  # in log10 Msun
    alpha = -1.25
    himf_pars = [phi_star,m_star,alpha]
    nhi,omegahi,psn = cal_himf(himf_pars,mmin,Planck18)
    assert np.allclose(nhi,0.13980687586146462)
    assert np.allclose(omegahi,0.00036495842278914405)
    assert np.allclose(psn,150.93927719814297)
    minput = np.linspace(mmin,11,500)
    nhi_cumu = cumu_nhi_from_himf(minput,mmin,himf_pars)
    assert np.allclose(nhi_cumu[0],0.0)
    assert np.allclose(nhi_cumu[-1],nhi)
    
def test_sample_from_dist():
    test_sample = sample_from_dist(uniform_pdf,0,1,size=1000000,cdf=False)
    count,_ = np.histogram(test_sample,bins=10)
    assert (np.abs((count-len(test_sample)/10)/(len(test_sample)/10))>0.01).sum() == 0
    test_sample = sample_from_dist(uniform_cdf,0,1,size=1000000,cdf=True)
    count,_ = np.histogram(test_sample,bins=10)
    assert (np.abs((count-len(test_sample)/10)/(len(test_sample)/10))>0.01).sum() == 0

def test_busy_function_simple():
    xarr = np.linspace(-10,10,101)
    assert np.allclose(busy_function_simple(xarr,2,1,0,0),erf(-xarr**2)+1)
    

def test_dft_mat():
    num_ch = np.random.randint(1,2000)
    xarr = np.random.normal(size=num_ch)
    f_arr = np.fft.fft(xarr)
    f_arr_test = dft_mat(num_ch) @ xarr
    assert np.allclose(f_arr_test,f_arr)

def test_busy_function_0():
    xarr = np.linspace(-10,10,101)
    assert np.allclose(busy_function_0(xarr,4,1,0,0),(erf(-xarr)+1)*(erf(xarr)+1))

def test_get_taper_renorm():
    window = np.ones(100)
    assert np.allclose(get_taper_renorm(window),1.0)

def test_unravel_list():
    inlist = [[1,2],[3,4]]
    assert unravel_list(inlist) == [1,2,3,4]

def test_cal_cov_simple():
    rand = np.random.normal(size=(1,100000))
    rand = np.vstack((rand,-rand))
    cov_test = cal_cov_simple(rand)
    cov_test /= np.sqrt(np.diagonal(cov_test))[:,None]*np.sqrt(np.diagonal(cov_test))[None,:]
    assert np.allclose(cov_test,[[1,-1],[-1,1]])