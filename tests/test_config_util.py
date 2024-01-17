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
                      'work_dir':'./'
                     }
    config = tidy_config_path(config)
    for val in list(config['FILE'].values()):
        assert val[-1]!='/'
    assert config['FILE']['LOG'] == os.getcwd()+'/'+'logs'
    assert config['FILE']['SCRIPT'] == os.getcwd()+'/'+'scripts'
    config = configparser.ConfigParser()
    config['FILE'] = {'1':'test/',
                      'LOG':'logs/',
                      'SCRIPT':'scripts/',
                      'work_dir':'./'
                     }
    config = tidy_config_path(config)
    assert config['FILE']['LOG'] == os.getcwd()+'/'+'logs'
    assert config['FILE']['SCRIPT'] == os.getcwd()+'/'+'scripts'