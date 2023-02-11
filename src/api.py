from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import util as utils
import data_pipeline as data_pipeline
import preprocessing as preprocessing

# load configuration
config = utils.load_config()
model_data = utils.pickle_load(config["production_model_path"])

class api_data(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    Age: int
    BMI: float
    DiabetesPedigreeFunction: float
    
app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):
    # convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)
    data.columns = config["predictors"]
    
    # convert data types
    data = pd.concat(
        [
            data[config["predictors"][:6]].astype(np.int64),
            data[config["predictors"][6:]].astype(np.float64)
        ],
        axis = 1
    )   
    # Check range data
    try:
        data_pipeline.check_data(data, config, True)  # type: ignore
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}

    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    if y_pred[0] == 0:
        y_pred = "Diindikasi tidak terkena penyakit diabetes."
    else:
        y_pred = "Diindikasi terkena penyakit diabetes."
    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)    