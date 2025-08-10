import os 
import pandas as pd
from src import logger
from sklearn.linear_model import ElasticNet
import joblib

from src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config
        
    def train(self):
        train_x = pd.read_csv(self.config.train_x_path)
        train_y = pd.read_csv(self.config.train_y_path)
        test_x = pd.read_csv(self.config.test_x_path)
        test_y = pd.read_csv(self.config.test_y_path)
        
        lr = ElasticNet(alpha=self.config.alpha, l1_ratio= self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
        