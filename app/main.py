import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from .schema import PredictionInput, PredictionResponse

app = FastAPI(
    title="Disease Prediction API",
    description="An API to predict diseases based on input features using a trained machine learning model.",
    version="1.0.0",
    contact={
        "name": "Ankit Singh",
        "email": "ankit21450@iiitd.ac.in",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Load the trained model and label encoder
model = joblib.load("app\data\Logistic_Regression_model.pkl")  
label_encoder = joblib.load("app\data\label_encoder.pkl")

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Disease",
    description="Predicts the disease based on the input features provided (132 features). The features must be binary of 132 entries and in the order the model was trained on.",
    operation_id="predict_disease"
)
async def predict(input_data: PredictionInput):
    """
    Predict the disease based on the input features.

    - **features**: List of binary feature values.
    - **response**: Returns the predicted class label as a string.
    """
    try:
        # Convert input list to numpy array
        feature_array = np.array(input_data.features).reshape(1, -1)

        # Perform prediction
        prediction = model.predict(feature_array)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return {"prediction": predicted_label}
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=f"ValueError: {str(ve)} - Please ensure the input features are valid and match the model's requirements.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
