
from fastapi import FastAPI
import joblib
import pandas as pd 
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)


model = joblib.load("../output/model.pkl")
col_transf = joblib.load("../output/column_transformer.joblib")
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO , 
format='%(asctime)s - %(levelname)s - %(message)s',
filename = 'logs/api.log', 
filemode = 'a')

logger = logging.getLogger(__name__)


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"Hello" : "World"}

@app.get("/health")
def read_status():
    logger.info("Health check endpoint accessed")
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):  
    logger.info("Prediction endpoint accessed")  
    try:
        input_data = pd.DataFrame([data])
        logger.info("Received data for prediction successfully")

        transformed_data = col_transf.transform(input_data)
        logger.info("Transformed data successfully")

        transformed_df = pd.DataFrame(transformed_data, columns=col_transf.get_feature_names_out())
        logger.info("DataFrame created successfully")

        predictions = model.predict(transformed_df)
        predictions = predictions.tolist()
        logger.info("Predictions made successfully")
        logger.info("Prediction result: %s", predictions[0])

        if predictions[0] == 0 :
            predictions[0] = "Not Churn"
        else:
            predictions[0] = "Churn"
        return {"prediction": predictions[0]}    
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return {"error": "An error occurred during prediction."}