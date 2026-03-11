import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            features = preprocessor.transform(features)
            predicted = model.predict(features)
            return predicted
        except Exception as e:
            raise CustomException(e, sys)
        