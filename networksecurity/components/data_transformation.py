import sys 
import os 
import numpy  as np
import pandas as pd 
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline 

from networksecurity.constant.training_pipeline.training_pipeline import TARGET_COLUM
from networksecurity.constant.training_pipeline.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object 


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact= data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
        
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
        
    def get_data_transformer_object(cls)->Pipeline:
        logging.info("Enterd get_data_tranformer_object method of Transformation class")
        try:
            imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Intialize KNNImputer with{DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor:Pipeline=Pipeline([{"imputer",imputer}])
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
        
        
        
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_dat_transformation method of DataTransformation class")
       
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
    
            ##trainning data frame 
            input_features_train_df = train_df.drop(columns=[TARGET_COLUM],axis=1)
            target_features_train_df= train_df[TARGET_COLUM]
            target_features_train_df = target_features_train_df.replace(-1,0)
            
            ##testing data frame 
            input_features_test_df = test_df.drop(columns=[TARGET_COLUM],axis=1)
            target_features_test_df= test_df[TARGET_COLUM]
            target_features_test_df = target_features_test_df.replace(-1,0)
            
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_features_train_df)
            transformed_input_train_feature= preprocessor_object.transform(input_features_train_df)
            transformed_input_test_feature= preprocessor_object.transform(input_features_test_df)
            
            
            train_arr = np.c_[transformed_input_train_feature,np.array(target_features_train_df)]
            test_arr = np.c_[transformed_input_test_feature,np.array(target_features_test_df)]
            
            ##save numpy array data 
            # Save numpy array data
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # Save the preprocessor object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object
            )
            
            ## preaparing artifacts 
            
            data_transformation_artifacts = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise NetworkSecuirtyException(e,sys)