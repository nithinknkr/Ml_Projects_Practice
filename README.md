# Student Performance Prediction

This repository contains a machine learning pipeline for predicting student math scores based on demographic and academic features.

## Features
- Data ingestion, transformation, model training and evaluation
- Uses scikit-learn, XGBoost, CatBoost, etc.
- Flask web app for running predictions and retraining the model

## Setup

1. Create and activate a Python virtual environment (the repository already includes a `env/` directory for a venv):
   ```powershell
   & "env\Scripts\Activate.ps1"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Run the training pipeline manually or through the Flask endpoint:

```bash
python -c "from src.components.data_ingestion import DataIngestion; from src.components.data_transformation import DataTransformation; from src.components.model_trainer import ModelTrainer;\
-ingestion = DataIngestion(); train_path,test_path = ingestion.initiate_data_ingestion();\
-transformer = DataTransformation(); train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path);\
-trainer = ModelTrainer(); print(trainer.initiate_model_trainer(train_arr, test_arr))"
```

or start the web server and visit `http://localhost:5000/train`.

## Using the Flask App

Start the application with:

```bash
python app.py
```

Then open a browser and navigate to `http://localhost:5000/`. Fill out the form with student information and click **Predict** to see the predicted math score. You can retrain the model using the **Retrain model** link.


## Project Structure

```
[src/ ...]
app.py            # flask application
requirements.txt  # python dependencies
templates/        # HTML templates for the web UI
artifacts/        # saved models and transformers
```