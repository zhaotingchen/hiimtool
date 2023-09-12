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
    
    #get absolute path if work_dir is just ./
    if config['FILE']['work_dir'] == '.':
        config['FILE']['work_dir'] = os.getcwd()
    
    #append the dirs
    for key in config['OUTPUT'].keys():
        if config['OUTPUT'][key][0]=='/':
            slash=''
        else:
            slash='/'
        config['OUTPUT'][key] = config['FILE']['work_dir']+slash+config['OUTPUT'][key]
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
    file_setup['args'] = first_line[2]
    file_setup['loop'] = first_line[3]
    return file_setup

def gen_syscall(calltype,
                script,
                config,
                args='',
                jobtype='',
                loop=0,
               ):
    '''
    Create a command for running a slurm job.
    Adapted from oxcat.
    '''
    sif_exec = 'singularity exec '
    if calltype == 'container':
        syscall = sif_exec+config['FILE']['container']+' python '+script+' '
        if loop >0:
            syscall_tot=''
            for i in range(loop):
                syscall_tot += syscall + str(i)+' '+args + ' \n'
        else:
            syscall_tot = syscall + args
        return syscall_tot
    if calltype == 'mpicasa':
        syscall = config['FILE']['mpicasa']+ ' singularity exec '
        syscall = syscall + config['FILE']['casa']
        syscall = syscall + ' casa --log2term --nogui -c '
        syscall = syscall + script + ' '
        if loop >0:
            syscall_tot=''
            for i in range(loop):
                syscall_tot += syscall + str(i)+' '+args + ' \n'
        else:
            syscall_tot = syscall + args
        return syscall_tot
    if calltype == 'env':
        syscall = 'source ' + config['FILE']['bash'] + ' \n'
        syscall = syscall + 'source activate ' + config['FILE']['env']
        syscall = syscall + ' \n'
        syscall = syscall + 'python ' + script + ' ' 
        if loop >0:
            syscall_tot=''
            for i in range(loop):
                syscall_tot += syscall + str(i)+' '+args + ' \n'
        else:
            syscall_tot = syscall + args
        return syscall_tot
    if calltype == 'envarray':
        syscall = 'source ' + config['FILE']['bash'] + ' \n'
        syscall = syscall + 'source activate ' + config['FILE']['env']
        syscall = syscall + ' \n'
        syscall = syscall + 'python ' + script + ' '
        syscall = syscall + '$SLURM_ARRAY_TASK_ID ' + config['SLURM_'+jobtype]['ARRAY']
        syscall = syscall + ' ' + args
        return syscall
    if calltype=='envmpi':
        num_core = int(config['SLURM_'+jobtype]['ntasks']*int(config['SLURM_'+jobtype]['CPUS']))
        syscall = 'source ' + config['FILE']['bash'] + ' \n'
        syscall = syscall + 'source activate ' + config['FILE']['env']
        syscall = syscall + ' \n'
        syscall += 'module load ' + config['FILE']['mpimod'] + ' \n'
        syscall += 'mpirun --mca btl vader,self -n '+str(num_core)+' python '+ script + ' ' + args
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
    if 'ARRAY' in slurm_config.keys():
        slurm_array = slurm_config['ARRAY']
        slurm_array = '#SBATCH --array=0-'+str(int(slurm_array)-1)+' \n'
    else:
        slurm_array = ''
    
    slurm_runfile = config['OUTPUT']['script']+'/slurm_'+jobname+'.sh'
    slurm_logfile = config['OUTPUT']['log']+'/slurm_'+jobname+'.log'
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
        slurm_array,
        'SECONDS=0\n',
        syscall+'\n',
        'echo "****ELAPSED "$SECONDS"s '+jobname+'"\n'])
#           'sleep 10\n'])
    f.close()
    #os.chmod(slurm_runfile, 0o444)
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
                file.write(section+'_'+key.replace('-','_')+' = \''+val+'\'\n')
    return 1

def gen_syscall_wsclean(msfile,
                config,
                file_setup,
               ):
    '''
    Create a command for running a slurm job.
    Adapted from oxcat.
    '''
    sif_exec = 'singularity exec '
    syscall = sif_exec + config['FILE']['wsclean']+' wsclean '
    for key,val in file_setup.items():
        syscall += '-'+key + ' ' + val + ' '
    syscall += msfile + ' \n'
    return syscall