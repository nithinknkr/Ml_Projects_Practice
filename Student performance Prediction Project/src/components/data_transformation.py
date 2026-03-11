import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        #this function is used to transform the data
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns =['gender' , 'race_ethnicity' , 'parental_level_of_education' , 'lunch' , 'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('one_hot_encoder', OneHotEncoder()),
                        ('scaler', StandardScaler(with_mean=False))
                    ]
            )
            logging.info(f"Categorical columns encoding Completed: {categorical_columns}")
            logging.info(f"Numerical columns Scaling Completed: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', categorical_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self , train_data_path, test_data_path ):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("read train and test data completed"
                         "initiating the data transformation")
            #selecting the input and target feature from the dataframe
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns =['gender' , 'race_ethnicity' , 'parental_level_of_education' , 'lunch' , 'test_preparation_course']
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            logging.info("selected input and target features from train and test data"                         "preprocessing object is being created")
            preprocessing_obj = self.get_data_transformer_object()
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(train_df[target_column_name])]
            test_arr = np.c_[input_feature_test_arr, np.array(test_df[target_column_name])]
            logging.info("saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
        