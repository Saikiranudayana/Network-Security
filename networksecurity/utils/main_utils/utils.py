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

        
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")

        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)

    except Exception as e:
        raise NetworkSecuirtyException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecuirtyException(e, sys) from e

import sys
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# If you're using a custom exception:
# from networksecurity.exception.exception import NetworkSecurityException

def evaluate_models(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    models: Dict[str, Any], 
    params: Dict[str, dict]
) -> Dict[str, float]:
    """
    Performs GridSearchCV for each model in 'models' with the corresponding 
    hyperparameters in 'params', fits the best estimator, and evaluates it.

    :param X_train: Training features
    :param y_train: Training labels
    :param X_test:  Test features
    :param y_test:  Test labels
    :param models:  Dictionary of model name -> model instance
    :param params:  Dictionary of model name -> parameter grid for that model
    :return:        Dictionary of model name -> test accuracy score
    """
    try:
        report = {}

        for model_name, model in models.items():
            # Retrieve hyperparameters for this specific model
            param_grid = params[model_name]

            # Set up GridSearchCV
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1  # Increase verbosity if you want more output
            )

            # Fit GridSearchCV on the training set
            gs.fit(X_train, y_train)

            # Extract the best model and retrain on full training data
            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            # Evaluate on the test set
            y_test_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Store the test accuracy in the report
            report[model_name] = test_accuracy

        return report

    except Exception as e:
        # Replace 'Exception' below with your custom exception if needed
        raise Exception(f"Error in evaluate_models: {e}") from e
        # or if you have a custom exception:
        # raise NetworkSecurityException(e, sys) from e
