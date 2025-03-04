from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.components.data_validation import DataValidation
import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Intiate the data ingestion process")
        dataingestionartifact =data_ingestion.inititate_data_ingestion()
        logging.info("Data initiation compoleted")
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation= DataValidation(dataingestionartifact,data_validation_config)
        logging.info("intitate the data validation")
        data_validation_artifact =data_validation.initiate_data_validation()
        logging.info("data validatioin completed")
       
    except Exception as e:
        raise NetworkSecuirtyException(e,sys)             