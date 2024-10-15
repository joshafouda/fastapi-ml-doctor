from huggingface_hub import hf_hub_download
import joblib
from typing import Annotated
from pydantic import create_model
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from app.utils import symptoms_list

ml_model = {}
REPO_ID = "AWeirdDev/human-disease-prediction"
FILENAME = "sklearn_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load the model once when the app starts
        ml_model["doctor"] = joblib.load(
            hf_hub_download(
                repo_id=REPO_ID, filename=FILENAME
            )
        )
        yield  # Makes the model available during the app's lifecycle
    finally:
        ml_model.clear()  # Clears the model when the app is shutting down


app = FastAPI(title="AI Doctor", lifespan=lifespan)


query_parameters = {
    symp: (bool, False)
    for symp in symptoms_list[:10]  # Limiting to the first 10 symptoms
}

Symptoms = create_model("Symptoms", **query_parameters)


@app.get("/diagnosis")
async def get_diagnosis(
    symptoms: Annotated[Symptoms, Depends()],
):
    # Convert the boolean inputs to a numerical array for the model
    array = [
        int(value)
        for _, value in symptoms.model_dump().items()
    ]
    
    # Ensure the array matches the input shape expected by the model
    array.extend([0] * (len(symptoms_list) - len(array)))

    try:
        # Get the prediction from the model
        diseases = ml_model["doctor"].predict([array])
        return {
            "diseases": [disease for disease in diseases]
        }
    except Exception as e:
        # In case of an error, return an appropriate HTTP response
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
