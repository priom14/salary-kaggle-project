import os
import numpy as np
import pandas as pd
from src import logger 
from src.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
 
class DataTransfroamtion:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def create_train_test_sets(self):
        data = pd.read_csv(self.config.data_dir)
        
        data = data.drop(columns = ['salary', 'salary_currency'])

        # Convert the 'work_year' column into a simpler format for the model
        data['work_year'] = data['work_year'] - 2020


        X = data.drop('salary_in_usd', axis=1)
        y = data['salary_in_usd']

        for col in ['job_title', 'employee_residence', 'company_location']:
            top_10 = X[col].value_counts().nlargest(10).index
            X[col] = np.where(X[col].isin(top_10), X[col], 'Other')

        # Initial Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        numerical_features = ['work_year', 'remote_ratio']
        ordinal_features = ['experience_level', 'company_size']
        nominal_features = ['employment_type', 'job_title', 'employee_residence', 'company_location']
        
        experience_order = ['EN', 'MI', 'SE', 'EX']
        size_order = ['S', 'M', 'L']
        
        
        preprocessor = ColumnTransformer(
        transformers=[
                # Transformer for numerical features
                ('num', StandardScaler(), numerical_features),

                # Transformer for ordinal features
                ('ord', OrdinalEncoder(categories=[experience_order, size_order]), ordinal_features),

                # Transformer for nominal features
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_features)
            ],
            remainder='passthrough'
                                        )       
        
        preprocessor.fit(X_train)
        
        X_train_transformed = preprocessor.transform(X_train)
        logger.info(f"X_train shape: {X_train_transformed.shape}")
        
        X_test_transformed = preprocessor.transform(X_test)
        logger.info(f"X_test shape: {X_test_transformed.shape}")
        
        
        feature_names = preprocessor.get_feature_names_out()
        
        # logger.info(f"Type of features: {type(feature_names)} & Features: {feature_names}")

        # Convert the transformed arrays back to DataFrames with proper column names
        X_train_df = pd.DataFrame(X_train_transformed.toarray(), columns=feature_names)
        X_test_df = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)

        # Save the DataFrames and Series to CSV files
        X_train_df.to_csv(os.path.join(self.config.root_dir,"train_x.csv"), index=False)
        X_test_df.to_csv(os.path.join(self.config.root_dir,"test_x.csv"), index=False)
        y_train.to_csv(os.path.join(self.config.root_dir,"train_y.csv"), index=False)
        y_test.to_csv(os.path.join(self.config.root_dir,"test_y.csv"), index=False)
        
        logger.info("Train test split completed")