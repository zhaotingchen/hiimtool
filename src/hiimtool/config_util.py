import glob
import datetime
import time
import os
import os.path as o
import sys
import configparser
import json

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def tidy_config_path(config):
    """
    Tidy up the paths in the config file.
    """
    #delete trailing slash
    for key in list(config['FILE'].keys()):
        if config['FILE'][key][-1]=='/':
            config['FILE'][key] = config['FILE'][key][:-1]
    
    #list of dir names to be appended to work_dir
    dirs_to_append = ('LOG','SCRIPT')
    
    #get absolute path if work_dir is just ./
    if config['FILE']['work_dir'] == '.':
        config['FILE']['work_dir'] = os.getcwd()
    
    #append the dirs
    for key in dirs_to_append:
        if config['FILE'][key][0]=='/':
            slash=''
        else:
            slash='/'
        config['FILE'][key] = config['FILE']['work_dir']+slash+config['FILE'][key]
    return config

def get_file_setup(file):
    '''
    read the type of set-up needed for a interim script
    '''
    file_setup = {}
    with open(file) as f:
        first_line = f.readline()
    first_line = first_line[1:-1]
    first_line = first_line.split(', ')
    file_setup['calltype'] = first_line[0]
    file_setup['jobtype'] = first_line[1]
    if len(first_line)>2:
        file_setup['args'] = first_line[2]
    else:
        file_setup['args'] = ''
    return file_setup

def gen_syscall(calltype,
                script,
                config,
                args='',
               ):
    '''
    Create a command for running a slurm job.
    Adapted from oxcat.
    '''
    sif_exec = 'singularity exec '
    if calltype == 'container':
        return sif_exec+config['FILE']['container']+' python '+script+' '+args
    if calltype == 'mpicasa':
        syscall = config['FILE']['mpicasa']+ ' singularity exec '
        syscall = syscall + config['FILE']['casa']
        syscall = syscall + ' casa --log2term --nogui -c '
        syscall = syscall + script + ' ' + args
        return syscall

def job_handler(syscall,
                jobname,
                config,
                jobtype,
                dependency = None,
                ):
    infrastructure = config['FILE']['infrastructure']
    if infrastructure != 'slurm':
        raise ValueError('Only slurm scheduler supported currently')
        
    slurm_config = config['SLURM_'+jobtype]
    slurm_time = slurm_config['TIME']
    slurm_partition = slurm_config['PARTITION']
    if slurm_partition != '':
        slurm_partition = '#SBATCH --partition='+slurm_partition+'\n'
    slurm_ntasks = slurm_config['NTASKS']
    slurm_nodes = slurm_config['NODES']
    slurm_cpus = slurm_config['CPUS']
    slurm_mem = slurm_config['MEM']
    
    slurm_runfile = config['FILE']['script']+'/slurm_'+jobname+'.sh'
    slurm_logfile = config['FILE']['log']+'/slurm_'+jobname+'.log'
    run_command = jobname+"=`sbatch "
    if dependency:
        #run_command += "-d afterok:${"+dependency+"} "
        run_command += '-d afterok:'+'${'+dependency.replace(':','}:${')+'} '
    run_command += slurm_runfile+" | awk '{print $4}'`"
    f = open(slurm_runfile,'w')
    f.writelines(['#!/bin/bash\n',
        '#file: '+slurm_runfile+':\n',
        '#SBATCH --job-name='+jobname+'\n',
        '#SBATCH --time='+slurm_time+'\n',
        slurm_partition,
        '#SBATCH --ntasks='+slurm_ntasks+'\n',
        '#SBATCH --nodes='+slurm_nodes+'\n',
        '#SBATCH --cpus-per-task='+slurm_cpus+'\n',
        '#SBATCH --mem='+slurm_mem+'\n',
        '#SBATCH --output='+slurm_logfile+'\n',
        'SECONDS=0\n',
        syscall+'\n',
        'echo "****ELAPSED "$SECONDS"s '+jobname+'"\n'])
#           'sleep 10\n'])
    f.close()
    #make_executable(slurm_runfile)
    run_command += '\n'

    return run_command

def ini_to_json(config_object):
    """
    Convert a configparser file to a dict for json dump
    """
    output_dict=dict()
    sections=config_object.sections()
    for section in sections:
        items=config_object.items(section)
        output_dict[section]=dict(items)
    
    return output_dict

def ini_to_py(config,filename):
    """
    Convert a configparser object to python file 
    """
    with open(filename, 'w') as file:
        file.write('')
    with open(filename, 'a') as file:
        for section in config.sections():
            for key,val in config[section].items(): 
                file.write(section+'_'+key+' = \''+val+'\'\n')
    return 1
