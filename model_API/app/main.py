import pickle
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
from utils.crete_time_series import create_time_series


class PredictionInput(BaseModel):
    pm10: float
    pm2_5: float
    pm2_5_O3_ratio: float


# Recurrent Neural Network Model
rnn_model = load_model('models/finalized_model.h5')
# Recurrent Neural Network Scaler
rnn_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

app = FastAPI()


@app.get("/health")
async def health():
    return {
        "status": "api is up and running",
    }


@app.post("/predict")
async def predict(data: List[PredictionInput]):
    window_size = 48

    data = [[data_slice.pm10, data_slice.pm2_5, data_slice.pm2_5_O3_ratio] for data_slice in data]

    scaled_data = rnn_scaler.transform(data)
    feature_cols = list(range(len(data[0])))

    X = create_time_series(scaled_data, window_size, feature_cols)

    prediction = rnn_model.predict(X)

    prediction_copies_array = np.repeat(prediction, len(feature_cols), axis=-1)

    prediction_reshaped = np.reshape(prediction_copies_array, (len(prediction), len(feature_cols)))

    prediction = rnn_scaler.inverse_transform(prediction_reshaped)[:, 0]

    return {"prediction": prediction.tolist()[0]}
