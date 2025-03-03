from networksecurity.exception.exception import NetworkSecuirtyException 
from networksecurity.logging.logger import logging 

##configuraTION OF THE DATA Ingestion Config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import  DataIngestionArtifact
 
import os
import sys 
import numpy as np 
import pandas as pd 
import pymongo 
from typing import List 
from sklearn.model_selection import train_test_split 

from dotenv import load_dotenv 

load_dotenv() 

MONGO_DB_URL =os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from MONGO DB
        """
        
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection= self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
                
            df.replace({"na":np.nan},inplace=True)
            return df 
        except Exception as e: 
            raise NetworkSecuirtyException
        
    def export_dat_into_feature_store(self,dataframe:pd.Dataframe):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_dir
            ##creating folder 
            dir_path = os.patyh.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
        
    def split_data_astrain_test(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set = train_test_split(
                dataframe,test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dat frame")
            
            logging.info(
                "Exited split_data_as_train_test method of Data_INgestion class"
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,index=False,header=True
            )
            
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,index=False,header=True
            )
            logging.info(f"Exported train and test file path")
 
            
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
        
    def inititate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe = self.export_dat_into_feature_store(dataframe)
            self.split_data_astrain_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(trained_file_path = self.data_ingestion_config.training_file_path,
                                                         test_file_path = self.data_ingestion_config.testing_file_path)

            return dataingestionartifact 
        except Exception as e: 
            raise NetworkSecuirtyException
            
