import pytest
import configparser
import sys
import os
from hiimtool.config_util import tidy_config_path

def test_tidy_config_path():
    config = configparser.ConfigParser()
    config['FILE'] = {'1':'test/',
                      'LOG':'logs',
                      'SCRIPT':'scripts',
                      'work_dir':'./',
                     }
    config['OUTPUT'] = {'test':'something'}
    config = tidy_config_path(config)
    for val in list(config['FILE'].values()):
        assert val[-1]!='/'
    for val in list(config['OUTPUT'].values()):
        assert val[-1]!='/'
    assert config['FILE']['work_dir'] == os.getcwd()
    assert config['OUTPUT']['test'] == os.getcwd()+'/'+'something'