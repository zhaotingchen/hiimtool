import sys
from casacore.tables import table
import re
import json
import glob
import numpy as np
import configparser 

hiimtool = '/idia/projects/mightee/zchen/hiimtool/src/'
sys.path.append(hiimtool)
from hiimtool.basic_util import vfind_scan,unravel_list,f_21,vfind_id
from hiimtool.ms_tool import read_ms,get_chanfreq
from hiimtool.grid_util import fill_row,grid_vis,worker_split,split_para_angle,worker,save_scan
from casatools import msmetadata,ms
from astropy import constants,units
import datetime
from multiprocessing import Pool
from subprocess import call
from itertools import repeat

fill=False # whether to fill the nearest neighbour
stokes = 'I'
col='corrected_data'
uvedges = np.linspace(-6000,6000,201)-30 # the edges of the u-v grids
num_ch = 220 #how many channels
flag_frac = 0.2
#scratch_dir = '/scratch3/users/ztchen/' # your scratch directory
scratch_dir = 'cache/' # your scratch directory

ms_dir = glob.glob('/idia/projects/hi_im/sourabh/DEEP2/1*')

freqarr1 = np.load('/idia/projects/mightee/zchen/freqarr.npy')
freqarr2 = np.load('/idia/projects/mightee/zchen/freqarr2.npy')

bin_id = '0.44'
if bin_id == '0.32':
    file_str = '/1*'
    freqarr = freqarr1.copy()
    save_dir = 'vis_grid/bin_1/'
else:
    file_str = '/9*'
    freqarr = freqarr2.copy()
    save_dir = 'vis_grid/bin_2/'
ms_list = ()
for ms_dir_i in ms_dir:
    ms_list += (glob.glob(ms_dir_i+file_str+'/selfcal')[0],)

block_list = vfind_id(ms_list)

chans = get_chanfreq(glob.glob(ms_list[0]+'/sub.mms/SUBMSS/*.ms')[0])
start_ch = np.where(chans==freqarr[0])[0][0]

sel_ch = [num_ch,start_ch,1,1] 


verbose=False
num_proc=15 # number of cores to use


#for file_indx in range(1,len(block_id)): # if you want to loop every time block
#for file_indx in range(8,len(block_id)): 
for file_indx in range(len(block_list)): 
    print('Reading block', block_list[file_indx], datetime.datetime.now().time().strftime("%H:%M:%S"))
    # get all the scans
    submslist = np.array(glob.glob(ms_list[file_indx]+'/sub.mms/SUBMSS/*.ms'))
    # get the scan id
    scan_id = vfind_scan(submslist)
    # initialise the grids   
    vis_even = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    vis_odd = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1),dtype='complex')
    if fill:
        count_even = np.zeros((len(uvedges)-1,len(uvedges)-1))
        count_odd = np.zeros((len(uvedges)-1,len(uvedges)-1))
    else:
        count_even = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
        count_odd = np.zeros((num_ch,len(uvedges)-1,len(uvedges)-1))
    # this part does the parallel computing
    if __name__ == '__main__':
        with Pool(num_proc) as p:
            p.starmap(save_scan, zip(range(len(scan_id)), repeat(
                (submslist,uvedges,flag_frac,block_list[file_indx],scratch_dir,sel_ch,stokes,col,fill,verbose,False))))
    # sum over the outputs
    for scan in scan_id:
        vis_i = np.load(scratch_dir+'vis_'+block_list[file_indx]+'_'+scan+'_'+stokes+'.npy')
        count_i = np.load(scratch_dir+'count_'+block_list[file_indx]+'_'+scan+'_'+stokes+'.npy')
        if int(scan)%2 == 0:
            vis_even+=vis_i
            count_even+=count_i
        else:
            vis_odd+=vis_i
            count_odd+=count_i
    # if you want to save the outputs for each timeblock
    # uncomment this if you loop over every timeblock to save the progress
    np.save(save_dir+'vissum_'+block_list[file_indx]+'_fill_'+str(fill)+'_odd_'+stokes,vis_odd)
    np.save(save_dir+'count_'+block_list[file_indx]+'_fill_'+str(fill)+'_odd_'+stokes,count_odd)
    np.save(save_dir+'vissum_'+block_list[file_indx]+'_fill_'+str(fill)+'_even_'+stokes,vis_even)
    np.save(save_dir+'count_'+block_list[file_indx]+'_fill_'+str(fill)+'_even_'+stokes,count_even)
    # remove the temporary files
    for scan in scan_id:
        vis_file = scratch_dir+'vis_'+block_list[file_indx]+'_'+scan+'_'+stokes+'.npy'
        count_file = scratch_dir+'count_'+block_list[file_indx]+'_'+scan+'_'+stokes+'.npy'
        call(["rm", vis_file])
        call(["rm", count_file])
    print('Block finished', block_list[file_indx], datetime.datetime.now().time().strftime("%H:%M:%S"))

