from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    features: list = Field(
        ..., 
        description="List of binary feature values (in the same order used for model training)."
    )

class PredictionResponse(BaseModel):
    prediction: str = Field(
        ..., 
        description="Predicted class label based on the input features."
    )