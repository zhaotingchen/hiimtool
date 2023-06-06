from hiimtool.basic_util import p2dim,chisq,vfind_scan,vfind_id,Specs
import pytest
import numpy as np

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