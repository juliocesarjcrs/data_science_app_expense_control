from .models import   PredictionRequest
from .utils import get_model
import pandas as pd

def get_prediction(request: PredictionRequest) -> dict:
    """
    Given a PredictionRequest object, this function returns a dictionary of predicted values.

    Args:
    - request: A PredictionRequest object with the following attributes:
        - steps: An integer representing the number of future time steps to predict.
        - date: A string representing the date at which to start making predictions.

    Returns:
    - A dictionary containing the predicted values for each time step specified in the request.
    """
    try:
        best_model = get_model()
    except Exception as e:
        return {"error": str(e)}
    # Set some variables needed for predictions
    len_test = 9
    periodo_ajust = len_test + request.steps
    data_start_test= '2022-07-31' # o la fecha + 1 mes en que terminó el entrenameinto
    future_dates = pd.date_range(start= data_start_test, periods=periodo_ajust, freq='M') # M month end frequency
    # Set some variables needed for predictions
    exogen_data = []
    for date in future_dates:
        exogen_data.append(date.days_in_month)

    # Make predictions using the model
    future_predictions = best_model.forecast(steps=periodo_ajust, exog=exogen_data)
    last_predictions = future_predictions[-request.steps:]

    # Format the predicted values as a dictionary and return it
    format_pred =last_predictions.to_dict()
    return format_pred