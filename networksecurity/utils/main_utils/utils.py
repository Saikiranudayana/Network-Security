import yaml
from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
import os
import sys 
import numpy as np 
import dill 
import pickle  


def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecuirtyException(e,sys)
    
def write_yaml_file(file_path:str,content: object,replace:bool = False)->None:
    try:
        if replace:
           if os.path.exists(file_path):
               os.remove(file_path) 
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecuirtyException(e,sys)

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to a file.
    
    :param file_path: str -> Location of the file to save
    :param array: np.array -> The NumPy array data to be saved
    """
    try:
        # Create parent directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the array in binary format
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecuirtyException(e,sys) from e
    

def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using pickle.

    :param file_path: str -> The path to the file where the object will be saved
    :param obj: object -> The Python object to be serialized and saved
    """
    try:
        logging.info("Entered the save_object method of MainUtils class")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Serialize and save the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Exited the save_object method of MainUtils class")

    except Exception as e:
        # Wrap any exception in a custom NetworkSecurityException
        raise NetworkSecuirtyException(e, sys) from e

        
