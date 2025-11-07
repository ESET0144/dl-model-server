import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF logs

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib


# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="template")


# Load the model and preprocessor
try:
    model = tf.keras.models.load_model("model.h5")
    preprocessor = joblib.load('hotel_preprocessor.pkl')
    print("✓ Model and preprocessor loaded successfully")
except Exception as e:
    print(f"❌ Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

# Define your feature groups
features_num = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr"
]

features_cat = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"
]

# Month mapping
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    hotel: str = Form(...),
    lead_time: float = Form(...),
    arrival_date_year: int = Form(...),
    arrival_date_month: str = Form(...),
    arrival_date_week_number: int = Form(...),
    arrival_date_day_of_month: int = Form(...),
    stays_in_weekend_nights: int = Form(...),
    stays_in_week_nights: int = Form(...),
    adults: int = Form(...),
    children: float = Form(...),
    babies: int = Form(...),
    meal: str = Form(...),
    market_segment: str = Form(...),
    distribution_channel: str = Form(...),
    is_repeated_guest: int = Form(...),
    previous_cancellations: int = Form(...),
    previous_bookings_not_canceled: int = Form(...),
    reserved_room_type: str = Form(...),
    deposit_type: str = Form(...),
    customer_type: str = Form(...),
    adr: float = Form(...),
    required_car_parking_spaces: int = Form(...),
    total_of_special_requests: int = Form(...)
):
    if model is None or preprocessor is None:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": "Model or preprocessor not loaded properly"
        })
    
    try:
        # Create a DataFrame with the input data
        input_data = {
            'lead_time': [lead_time],
            'arrival_date_year': [arrival_date_year],
            'arrival_date_week_number': [arrival_date_week_number],
            'arrival_date_day_of_month': [arrival_date_day_of_month],
            'stays_in_weekend_nights': [stays_in_weekend_nights],
            'stays_in_week_nights': [stays_in_week_nights],
            'adults': [adults],
            'children': [children],
            'babies': [babies],
            'is_repeated_guest': [is_repeated_guest],
            'previous_cancellations': [previous_cancellations],
            'previous_bookings_not_canceled': [previous_bookings_not_canceled],
            'required_car_parking_spaces': [required_car_parking_spaces],
            'total_of_special_requests': [total_of_special_requests],
            'adr': [adr],
            'hotel': [hotel],
            'arrival_date_month': [month_mapping[arrival_date_month]],
            'meal': [meal],
            'market_segment': [market_segment],
            'distribution_channel': [distribution_channel],
            'reserved_room_type': [reserved_room_type],
            'deposit_type': [deposit_type],
            'customer_type': [customer_type]
        }

        df_input = pd.DataFrame(input_data)

        X_processed = preprocessor.transform(df_input)

        # Make prediction
        prediction_proba = model.predict(X_processed, verbose=0)
        probability = float(prediction_proba[0][0])

        # Determine result
        if probability > 0.5:
            prediction = "Likely to Cancel"
            confidence = probability
        else:
            prediction = "Not Likely to Cancel"
            confidence = 1 - probability


        return templates.TemplateResponse("result.html", {
                "request": request,
                "prediction": prediction,
                "confidence": round(confidence * 100, 2),
                "probability": round(probability * 100, 2)
            })

    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Prediction error: {str(e)}"
        })



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
