import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error , root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self , train_array , test_array):
        try:
            logging.info("Splitting the training and test input data")
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Extra Trees Regressor": ExtraTreesRegressor()
            }
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return f"Model Name: {best_model_name} , R2 Score: {r2_square}"
        except Exception as e:
            raise CustomException(e, sys)