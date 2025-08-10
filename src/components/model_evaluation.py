import os 
import pandas as pd
import numpy as np
from src import logger
from sklearn.metrics import r2_score, mean_absolute_error
from src.constants import *
from src.utils.common import read_yaml, create_directories, save_json
from urllib.parse import urlparse
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn


from src.entity.config_entity import ModelEvaluationConfig


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/mepriom01/salary-kaggle-project.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'mepriom01'
os.environ['MLFLOW_TRACKING_PASSWORD'] = "8fdf119fba6fd2ec94adccdb5b9ca2315b5c378d"


class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, actual, pred):
        mse = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return r2, mse
    
    def log_into_mlflow(self):
        test_x = pd.read_csv(self.config.test_x_path)
        test_y = pd.read_csv(self.config.test_y_path)
        model = joblib.load(self.config.model_path)
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            
            predictions = model.predict(test_x)
            r2, mse = self.eval_metrics(test_y, predictions)
            
            scores = {'mse': mse, 'r2': r2}
            save_json(path = Path(self.config.metric_file_name), data = scores)
            
            mlflow.log_params(self.config.all_params)
            
            mlflow.log_metric('mse', mse)
            mlflow.log_metric('r2', r2)
            
            if tracking_url_type_store != 'file':
                mlflow.log_artifact(self.config.model_path, artifact_path="model")

                # Step 2: Register the model from the artifact path.
                # We construct the URI to point to the artifact we just logged.
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"
                mlflow.register_model(model_uri=model_uri, name="ElasticNet")
            else:
                mlflow.sklearn.log_model(model, "model")
        