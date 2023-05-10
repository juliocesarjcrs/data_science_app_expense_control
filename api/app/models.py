from pydantic import BaseModel, Field
from datetime import date

class PredictionRequest(BaseModel):
    date: date
    # date: date = Field(default_factory=date.today, description='Start date for the prediction')
    # date: str = Field('2023-05-01', description='Fecha de inicio para la predicción')
    exogenus: list = Field([31, 30, 31, 31, 30], description='Datos exógenos para la predicción')
    steps: int = Field(3, description='Número de pasos a predecir')

class PredictionResponse(BaseModel):
    predcit: dict


    # class PredictionRequest(BaseModel):
#     date: date = Field(default=date.today(), description='Start date for the prediction')
#     exogenus: List[int] = Field([31, 30, 31, 31, 30], description='Exogenous data for the prediction')
#     steps: int = Field(5, description='Number of steps to predict')