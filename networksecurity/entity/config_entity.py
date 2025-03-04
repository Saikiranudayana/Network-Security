from datetime import datetime 
import os 
from networksecurity.constant.training_pipeline import  training_pipeline




print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)  



class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifacts_name = training_pipeline.ARTIFACT_DIR
        self.artifacts_dir = os.path.join(self.artifacts_name,timestamp)
        self.timestamps:str = timestamp
    
class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifacts_dir,training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_dir:str = os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR
        )   
        self.training_file_path:str = os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.FILE_NAME
        )   
        self.testing_file_path:str = os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.TEST_FILE_NAME
        )
        self.train_test_split_ratio:float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name:str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name:str = training_pipeline.DATA_INGESTION_DATABASE_NAME
        
class DataValidationConfig:
    def __int__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str = os.path.join(
            training_pipeline_config.artifacts_dir,training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir:str = os.path.join(
            self.data_validation_dir,training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir:str = os.path.join(
            self.data_validation_dir,training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path:str = os.path.join(
            self.valid_data_dir,training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path:str = os.path.join(    
            self.valid_data_dir,training_pipeline.TEST_FILE_NAME
        )
        self.invalid_train_file_path:str = os.path.join(
            self.invalid_data_dir,training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path:str = os.path.join(
            self.invalid_data_dir,training_pipeline.TEST_FILE_NAME
        )   
        
        self.drift_report_file_path:str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )
        
        

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_ARTIFACT_DIR
        )

        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRAIN_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )

        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TEST_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )

        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory where model trainer artifacts will be stored
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )

        # Path for saving the trained model file
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_FILE_NAME
        )

        # Expected accuracy (or score) threshold for the trained model
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE

        # Overfitting/underfitting threshold for model performance
        self.overfitting_underfitting_threshold: float = (
            training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        )
