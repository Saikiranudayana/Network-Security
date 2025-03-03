import yaml
from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
import os,sys 
import numpy as np 
import dill 
import pickle  


def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecuirtyException(e,sys)