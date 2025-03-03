from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info("Intiate the data ingestion process")
        dataingestionartifact =dataingestion.inititate_data_ingestion()
        print(dataingestionartifact)
       
    except Exception as e:
        raise NetworkSecuirtyException(e,sys)             