import os
import sys

from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metrics import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow


class ModedlTrainer:
    def __int__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
        
        
        
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score
            
            mlflow.log_metric("f1 score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
            
            
            
            
    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }
        params = {
        "Decision Tree": {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_features": ["sqrt", "log2", None],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [8, 16, 32, 64, 128, 256],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_features": ["sqrt", "log2", None],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "Gradient Boosting": {
            "learning_rate": [0.1, 0.05, 0.01],
            "subsample": [0.6, 0.8, 1.0],
            "n_estimators": [50, 100, 200],
            "criterion": ["friedman_mse", "squared_error"],
            "max_depth": [3, 5, 7]
        },
        "Logistic Regression": {
            "solver": ["lbfgs", "liblinear", "saga"],
            "C": [0.01, 0.1, 1, 10, 100],
            "max_iter": [100, 200, 500]
        },
        "AdaBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
            "algorithm": ["SAMME", "SAMME.R"]
        }
        }
        
        model_report:dict=evaluate_models(x_train=X_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                           models=models,param=params)
        
        ##to get best model score from dict
        best_model_score= max(sorted(model_report.values()))
        
        ##To get best model name from dict
        
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        
        y_train_pred =best_model.predict(X_train)
        
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        
        ##Track the mlfow
        self.track_mlflow(best_model,classification_train_metric)
        
        
        
        y_test_pred = best_model.predict(x_test)
        classification_test_metrics = get_classification_score(y_true=y_test,y_pred=y_test_pred)
        
        self.track_mlflow(best_model,classification_test_metrics)
        preprocessor = load_object(file_path= self.data_transformation_artifact.transformed_object_file_path)
        
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        
        Network_Model = NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
        
        
        ##Model Trainer artifact
        
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metrics)
        
        logging.info(f"Model trainer artifacts:{model_trainer_artifact}")
        return model_trainer_artifact
        
        


        
        
            
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            ##Loading trainnng array and testing array
            train_arr= load_numpy_array_data(train_file_path)
            test_arr= load_numpy_array_data(test_file_path)
            
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            
            model_trainer_artifact = self.train_model(x_train,y_train)
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)