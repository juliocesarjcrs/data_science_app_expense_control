from fastapi import FastAPI, HTTPException
from typing import Union
from datetime import datetime, timedelta
from .app.models import PredictionRequest,PredictionResponse
from .app.views import get_prediction
app = FastAPI()

@app.post('/v1/prediction')
def make_model_prediction(request: PredictionRequest):
     """
     Hace una predicción utilizando un modelo entrenado.

     - **date**: Fecha a partir de la cual se desea hacer la predicción Foramt= 'YYYY-MM-dd' "2023-04-28".
     - **exogenus**: Lista de valores exógenos para la predicción.
     - **steps**: Número de pasos a predecir a partir de la fecha proporcionada.
     """
     previous_month = request.date - timedelta(days=30)
     current_date = datetime.now()
     print('previous_month=', previous_month)
     print('current_date=', current_date)
     if current_date.month != previous_month.month + 1 or current_date.year != previous_month.year:
         raise HTTPException(status_code=400, detail="Invalid date")
     if request.steps < 1 and request.steps > 12:
         raise HTTPException(status_code=400, detail='Step must be greater than 0 and less than 12')
     return PredictionResponse(predcit=get_prediction(request))



@app.get("/")
def read_root():
    return {"Hello": "World"}


