from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation

import sys 
from networksecurity.components.model_trainer import ModedlTrainer,ModelTrainerArtifact
from networksecurity.entity.artifact_entity import ModelTrainerArtifact
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
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        logging.info("Data transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data transformation completed")
        
        logging.info("Model Training started")
        model_trainer_config= ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainerConfig(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.iniate_model_trainer()
        logging.info("Model Trainng artiffact created")
    except Exception as e:
        raise NetworkSecuirtyException(e,sys) from e             