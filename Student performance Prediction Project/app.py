import os
import sys
from flask import Flask, render_template, request

# make sure project root is on path so that src package can be imported
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # collect form values
            features = {
                "gender": request.form.get("gender"),
                "race_ethnicity": request.form.get("race_ethnicity"),
                "parental_level_of_education": request.form.get("parental_level_of_education"),
                "lunch": request.form.get("lunch"),
                "test_preparation_course": request.form.get("test_preparation_course"),
                "writing_score": float(request.form.get("writing_score")),
                "reading_score": float(request.form.get("reading_score"))
            }

            # create a one-row DataFrame (pipeline expects pandas DataFrame)
            import pandas as pd
            input_df = pd.DataFrame([features])

            predictor = PredictPipeline()
            prediction = predictor.predict(input_df)
            # prediction is usually an array; grab single element for display
            if hasattr(prediction, "__len__"):
                prediction_value = prediction[0]
            else:
                prediction_value = prediction
            return render_template("result.html", prediction=prediction_value)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return render_template("result.html", prediction=f"Error: {e}")
    else:
        return render_template("index.html")

@app.route("/train")
def train():
    """Re-run the training pipeline (only for development/demo)."""
    try:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)
        trainer = ModelTrainer()
        result = trainer.initiate_model_trainer(train_arr, test_arr)
        return f"Training completed: {result}"
    except Exception as e:
        logging.error(f"Training error: {e}")
        return f"Error during training: {e}"


# additional API endpoint for programmatic use
@app.route("/predict", methods=["POST"])
def predict_api():
    """Accepts JSON body with the same fields as the form and returns prediction."""
    try:
        data = request.get_json()
        # convert to DataFrame
        import pandas as pd
        input_df = pd.DataFrame([data])
        predictor = PredictPipeline()
        prediction = predictor.predict(input_df)
        prediction_value = prediction[0] if hasattr(prediction, "__len__") else prediction
        return {"prediction": prediction_value}
    except Exception as e:
        logging.error(f"API prediction error: {e}")
        return {"error": str(e)}, 400

if __name__ == "__main__":
    # this will launch Flask development server
    app.run(host="0.0.0.0", port=5000, debug=True)
