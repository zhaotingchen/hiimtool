'''
Utility functions for analysing measurementset files.
'''

from casacore.tables import table
from casatools import ms
import numpy as np
import sys
import time
import glob
from astropy import constants,units
import datetime
import re
import pickle
import configparser
from astropy.coordinates import SkyCoord
from casacore import quanta,measures
from .basic_util import calcsep

meerkat_bands = [(815e6,1080e6,'UHF'),
    (856e6,1711e6,'L'),
    (1750e6,2624e6,'S0'),
    (1969e6,2843e6,'S1'), # 2406.25
    (2188e6,3062e6,'S2'), # 2625.00 1654978576
    (2406e6,3281e6,'S3'), # 2843.75
    (2625e6,3499e6,'S4')] # 3062.50 1653833475

def read_ms(filename,keys,sel_ch=None,verbose=False,ifraxis=False):
    '''
    A function to read in the measurementset data and returns the values.
    
    Parameters
    ----------
        filename: str 
            The name of the measurementset file to read.
        
        keys: list or array of strings 
            The keywords to read.
            
        sel_ch: float array
            The channel selection function to pass into :py:meth:`casatools.selectchannel`. 
            See https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms.ms.selectchannel .
        verbose: bool, default False
            Whether to print out time information.
        ifraxis: bool, default False
            Whether to shape the data in the shape of (num_time,num_bl).
            
    Returns
    -------
        data: a list of np arrays containing the data corresponding to the keywords, indexed according to the input `keys`.
    '''
    if verbose:
        print('Reading data...',datetime.datetime.now().time().strftime("%H:%M:%S"))
    msset = ms()
    msset.open(filename)
    if sel_ch is not None:
        msset.selectchannel(sel_ch[0],sel_ch[1],sel_ch[2],sel_ch[3])
    data = msset.getdata(keys)
    keylist = np.array(list(data.keys()))
    key_pos = np.where(np.array(keys)[:,None]==keylist[None,:])[-1]
    data_list = ()
    if ifraxis:
        timearr = msset.getdata('time')['time']
        num_time = np.unique(timearr).size
    msset.close()
    for data_value in list(data.values()):
        if ifraxis:
            if isinstance(data_value,np.ndarray):
                data_shape = np.array(data_value.shape)
                data_shape[-1] = num_time
                data_shape = np.append(data_shape,-1)
                data_value = data_value.reshape(data_shape)
        data_list += (data_value,)
    data = np.array(data_list,dtype='object')[key_pos]
    if verbose:
        print('Finished',datetime.datetime.now().time().strftime("%H:%M:%S"))
    return data

def get_nchan(master_ms):

    """ Returns the number of channels in master_ms.
    Only works for data with a single SPW.
    Taken from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    spw_table = table(master_ms+'/SPECTRAL_WINDOW',ack=False)
    nchan = spw_table.getcol('NUM_CHAN')[0]
    spw_table.close()
    return nchan

def get_band(master_ms):

    """ Returns the minimum and maxmium frequency
    and band estimate.
    Taken from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    spw_table = table(master_ms+'/SPECTRAL_WINDOW',ack=False)
    chans = spw_table.getcol('CHAN_FREQ')[0]
    min_freq = chans[0]
    max_freq = chans[-1]
    mid_freq = np.mean((min_freq,max_freq))
    bw = max_freq - min_freq
    spw_table.close()

    f0s = []
    for band in meerkat_bands:
        fc = np.mean((band[0],band[1]))
        f0s.append(fc)
    diffs = np.abs(f0s-mid_freq)
    idx = diffs.tolist().index(np.min(diffs))
    band = meerkat_bands[idx][2]

    return min_freq,mid_freq,max_freq,bw,band

def get_antnames(master_ms):

    """ Returns a list of the antenna names in master_ms
    Taken from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    ant_tab = table(master_ms+'/ANTENNA',ack=False)
    ant_names = ant_tab.getcol('NAME')
    ant_names = [a.lower() for a in ant_names]
    ant_tab.close()
    return ant_names

def get_chanfreq(master_ms):
    """ Returns the frequencies of the channels """
    spw_table = table(master_ms+'/SPECTRAL_WINDOW',ack=False)
    chans = spw_table.getcol('CHAN_FREQ')[0]
    return chans
    

def get_fields(master_ms):

    """ Returns lists of directions, names and integer source IDs
    from the FIELD table of master_ms
    Taken from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    field_tab = table(master_ms+'/FIELD',ack=False)
    field_dirs = field_tab.getcol('REFERENCE_DIR')*180.0/np.pi
    field_names = field_tab.getcol('NAME')
    field_ids = field_tab.getcol('SOURCE_ID')
    field_tab.close()
    return field_dirs,field_names,field_ids

def get_states(
    master_ms,
    primary_intent,
    secondary_intent,
    target_intent,
    polarization_intent=None,
):

    """ Provide the partial string matches for primary, secondary and target scan
    intents (optionally polarization calibrator intent) and the corresponding integer STATE_IDs are extracted from the STATE
    table, along with any UNKNOWN states.
    Modified from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    state_tab = table(master_ms+'/STATE',ack=False)
    modes = state_tab.getcol('OBS_MODE')
    state_tab.close()

    target_state = []
    primary_state = []
    secondary_state = []
    pol_state = []
    unknown_state = []
    for i in range(0,len(modes)):
        if modes[i] == target_intent:
            target_state.append(i)
        if primary_intent in modes[i]:
            primary_state.append(i)
        elif secondary_intent in modes[i]:
            secondary_state.append(i)
        elif polarization_intent is not None:
            if polarization_intent in modes[i]:
                pol_state.append(i)
        elif modes[i] == 'UNKNOWN':
            unknown_state.append(i)

    if polarization_intent is None:
        return primary_state, secondary_state, target_state, unknown_state
    else:
        return primary_state, secondary_state, target_state, unknown_state, pol_state

def get_primary_candidates(master_ms,
                primary_state,
                unknown_state,
                field_dirs,
                field_names,
                field_ids):

    """ Automatically identify primary calibrator candidates from master_ms 
    Modified from [oxcat](https://github.com/IanHeywood/oxkat).
    
    """

    candidate_ids = []
    candidate_names = []
    candidate_dirs = []

    main_tab = table(master_ms,ack=False)
    for i in range(0,len(field_ids)):
        field_dir = field_dirs[i]
        field_name = field_names[i]
        field_id = field_ids[i]
        sub_tab = main_tab.query(query='FIELD_ID=='+str(field_id))
        states = np.unique(sub_tab.getcol('STATE_ID'))
        for state in states:
            if state in primary_state or state in unknown_state:
                candidate_dirs.append(field_dir)
                candidate_names.append(field_name)
                candidate_ids.append(str(field_id))
        sub_tab.close()
    main_tab.close()

    return candidate_dirs, candidate_names, candidate_ids

def get_secondaries(master_ms,
                secondary_state,
                field_dirs,
                field_names,
                field_ids):

    """ Automatically identify secondary calibrators from master_ms 
    Copied from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    secondary_ids = []
    secondary_names = []
    secondary_dirs = []

    main_tab = table(master_ms,ack=False)
    for i in range(0,len(field_ids)):
        field_dir = field_dirs[i]
        field_name = field_names[i]
        field_id = field_ids[i]
        sub_tab = main_tab.query(query='FIELD_ID=='+str(field_id))
        states = np.unique(sub_tab.getcol('STATE_ID'))
        for state in states:
            if state == secondary_state:
                secondary_dirs.append(field_dir[0].tolist())
                secondary_names.append(field_name)
                secondary_ids.append(str(field_id))
        sub_tab.close()
    main_tab.close()

    return secondary_dirs, secondary_names, secondary_ids

def get_targets(master_ms,
                target_state,
                field_dirs,
                field_names,
                field_ids):

    """ Automatically identify secondary calibrators from master_ms
    Copied from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    target_ids = []
    target_names = []
    target_dirs = []

    main_tab = table(master_ms,ack=False)
    for i in range(0,len(field_ids)):
        field_dir = field_dirs[i]
        field_name = field_names[i]
        field_id = field_ids[i]
        sub_tab = main_tab.query(query='FIELD_ID=='+str(field_id))
        states = np.unique(sub_tab.getcol('STATE_ID'))
        for state in states:
            if state == target_state:
                target_dirs.append(field_dir[0].tolist())
                target_names.append(field_name)
                target_ids.append(str(field_id))
        sub_tab.close()
    main_tab.close()

    return target_dirs, target_names, target_ids

def get_polarizations(master_ms,
                pol_state,
                field_dirs,
                field_names,
                field_ids):

    """ Automatically identify polarization calibrators from master_ms
    Copied from [oxcat](https://github.com/IanHeywood/oxkat).
    """
    
    pol_ids = []
    pol_names = []
    pol_dirs = []

    main_tab = table(master_ms,ack=False)
    for i in range(0,len(field_ids)):
        field_dir = field_dirs[i]
        field_name = field_names[i]
        field_id = field_ids[i]
        sub_tab = main_tab.query(query='FIELD_ID=='+str(field_id))
        states = np.unique(sub_tab.getcol('STATE_ID'))
        for state in states:
            if state == pol_state:
                pol_dirs.append(field_dir[0].tolist())
                pol_names.append(field_name)
                pol_ids.append(str(field_id))
        sub_tab.close()
    main_tab.close()

    return pol_dirs, pol_names, pol_ids

def get_primary_tag(candidate_dirs,
                candidate_names,
                candidate_ids):

    """ Use a positional match to identify whether a source is 1934 or 0408 
    from a list of candidates. Manual model required for 0408, and different
    flux scale standards required in setjy for 0408 and everything else.
    Copied from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    # Tags and positions for the preferred primary calibrators
    preferred_cals = [('1934',294.85427795833334,-63.71267375),
        ('0408',62.084911833333344,-65.75252238888889)]

    primary_tag = ''
    primary_name = []
    primary_id = [] 
    primary_tag = [] 
    primary_sep = []

    for i in range(0,len(candidate_dirs)):
        candidate_dir = candidate_dirs[i][0]
        candidate_name = candidate_names[i]
        candidate_id = candidate_ids[i]

        for cal in preferred_cals:
            test_sep = calcsep(candidate_dir[0],candidate_dir[1],cal[1],cal[2])
            if test_sep < 3e-3:
                primary_name.append(candidate_name)
                primary_id.append(str(candidate_id))
                primary_tag.append(cal[0])
                primary_sep.append(test_sep)

    if primary_tag == '':
        primary_name = candidate_names[0]
        primary_id = str(candidate_ids[0])
        primary_tag = 'other'
        primary_sep = 0.0


    return primary_name,primary_id,primary_tag,primary_sep

def target_cal_pairs(target_dirs,target_names,target_ids,
                secondary_dirs,secondary_names,secondary_ids):
    """
    The target_cal_map is a list of secondary field IDs of length target_ids.
    It links a specific secondary to a specific target.
    Copied from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    # The target_cal_map is a list of secondary field IDs of length target_ids
    # It links a specific secondary to a specific target
    target_cal_map = []
    target_cal_separations = []

    for i in range(0,len(target_dirs)):
        ra_target = target_dirs[i][0]
        dec_target = target_dirs[i][1]
        separations = []
        for j in range(0,len(secondary_dirs)):
            ra_cal = secondary_dirs[j][0]
            dec_cal = secondary_dirs[j][1]
            separations.append(calcsep(ra_target,dec_target,ra_cal,dec_cal))
        separations = np.array(separations)
        secondary_index = np.where(separations == np.min(separations))[0][0]

        target_cal_map.append(str(secondary_names[secondary_index]))
        target_cal_separations.append(round(separations[secondary_index],3))

    return target_cal_map,target_cal_separations

def target_ms_list(working_ms,target_names):

    """ 
    Return a list of MS names derived from target_names.
    Copied from [oxcat](https://github.com/IanHeywood/oxkat).
    """

    target_ms = []
    for target in target_names:
        ms_name = working_ms.replace('.ms','_'+target.replace(' ','_')+'.ms')
        target_ms.append(ms_name)

    return target_ms

def get_nscan(master_ms):

    """ 
    Return the number of scans in the measurementset.
    """
    tab = table(master_ms,ack=False)
    num_scan = len(np.unique(tab.getcol('SCAN_NUMBER')))

    return num_scan

def get_secondaries(master_ms,
                secondary_state,
                field_dirs,
                field_names,
                field_ids):

    """ Automatically identify secondary calibrators from master_ms """

    secondary_ids = []
    secondary_names = []
    secondary_dirs = []

    main_tab = table(master_ms,ack=False)
    for i in range(0,len(field_ids)):
        field_dir = field_dirs[i]
        field_name = field_names[i]
        field_id = field_ids[i]
        sub_tab = main_tab.query(query='FIELD_ID=='+str(field_id))
        states = np.unique(sub_tab.getcol('STATE_ID'))
        for state in states:
            if state == secondary_state:
                secondary_dirs.append(field_dir[0].tolist())
                secondary_names.append(field_name)
                secondary_ids.append(str(field_id))
        sub_tab.close()
    main_tab.close()

    return secondary_dirs, secondary_names, secondary_ids

def get_para_angle(ant_pos,field_centre,time_step,unit='deg'):
    '''
    get parallalctic angle of an observation.
    Adapted from [Codex Africanus](https://github.com/ratt-ru/codex-africanus/blob/c823ef364e7b62e4d3ce527b8ccdec0b4115e3ff/africanus/rime/parangles.py).

    Parameters
    ----------
        ant_pos: casa record
            The position of the antenna. Can be read from `casatools.msmetadata.antennaposition`.

        field_centre: casa record
            The phase centre of the observation. Assumed to also be the pointing centre. Can be read from `casatools.msmetadata.phasecenter`.

        time_step: float.
            The time of the observation, in UTC scale and MJD second format.

        unit: str, default "deg".
            The unit of the angle returned. Can be "deg" or "rad".
    
    Returns
    -------
        para_angle: float.
            The parallactic angle of the observation in specificed unit.
    '''
    meas_serv = measures.measures()
    zenith = meas_serv.direction('AZELGEO', '0deg', '90deg')
    meas_serv.do_frame(meas_serv.epoch("UTC", quanta.quantity(time_step, "s")))
    meas_serv.do_frame(ant_pos)
    para_angle = meas_serv.posangle(field_centre, zenith).get_value(unit)
    return para_angle