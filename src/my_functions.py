import pandas as pd
import numpy as np
import holidays
import datetime
# metrics error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from pmdarima.metrics import smape
from tabulate import tabulate
from IPython.display import display, Markdown
# handle time
import locale
locale.setlocale(locale.LC_MONETARY, 'es_CO.UTF-8')

co_holidays = holidays.CO()
import matplotlib.pyplot as plt
import plotly.express as px
import os
import itertools
import statsmodels.api as sm
import plotly.graph_objects as go

#models
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

def num_holidays(start, end):

    range_of_dates = pd.date_range(start=start, end=end)
    df = pd.DataFrame(
        index=range_of_dates,
        data={"is_holiday": [date in co_holidays for date in range_of_dates]}
    )
    return df.sum()

def dict_month_num_holidays(start, end):

    daterange = pd.date_range(start, end, freq='1M')

    month_holiday_dict = {}

    for d in daterange:
        # end_month = date.replace(day = calendar.monthrange(date.year, date.month)[1])
        firstDayOfMonth = datetime.date(d.year, d.month, 1)
        sum_holidays  = num_holidays(firstDayOfMonth, d)
        new_data = {d.strftime("%Y-%m-%d"):sum_holidays.is_holiday}
        month_holiday_dict.update(new_data)
    holidays_by_month = pd.DataFrame(month_holiday_dict.items(), columns=['date', 'holidays_cont'])
    holidays_by_month['date'] = pd.to_datetime(holidays_by_month['date'], format="%Y-%m-%d")
    return holidays_by_month

def add_holidays_to_frame(resample_date, df_time):
    # Obtener el comienzo y fin de la fecha  de mis datos remuestreados
    start_date = resample_date.index[0]
    last = len(resample_date.index)
    last_date = resample_date.index[last-1]
    # Obtener dataframe con los números de festivos por cada mes
    holidays_by_month = dict_month_num_holidays(start_date, last_date)

    # unir los datos a mi data frame
    result = df_time.merge(holidays_by_month, how='left', on='date')
    result.set_index('date', inplace=True)
    return result

def is_first_or_last_month(date, is_first=False):
    """
    Determina si un mes es el primer mes o el último mes del año y devuelve un valor binario.

    Parameters:
        date (datetime): Fecha que representa el mes a evaluar.
        is_first (bool): Indica si se quiere determinar si es el primer mes (True) o el último mes (False). 
                         Por defecto, es False.

    Returns:
        int: Valor binario que indica si el mes es el primero o el último. 1 representa True, 0 representa False.

    """
    month = date.month
    return int(month == 1) if is_first else int(month == 12)

def resample_dataframe(df, new_freq='M'):
    """
    Resample el DataFrame según una nueva frecuencia y suma los valores de la columna 'cost'.

    Parameters:
        df (DataFrame): El DataFrame original.
        new_freq (str): La nueva frecuencia para resamplear el DataFrame. Por defecto es 'M' (mes).

    Returns:
        DataFrame: El DataFrame resampleado con la nueva frecuencia y la suma de 'cost'.

    """
    resample_date = df.resample(new_freq).cost.sum()
    resample_date.drop(resample_date.tail(1).index, inplace=True)
    return resample_date

def visualize_resampled_data(resampled_data):
    """
    Visualiza los datos resampleados en un gráfico de línea.

    Parameters:
        resampled_data (DataFrame): El DataFrame resampleado con la nueva frecuencia y la suma de 'cost'.

    Returns:
        None

    """
    plt.figure(figsize=(14, 5))
    fig = px.line(resampled_data, x=resampled_data.index, y="cost", title='Gastos en el tiempo')
    fig.show()

def feature_engineering(df):
    """
    Realiza ingeniería de características en el DataFrame, agregando columnas como 'month', 'year', 'days_in_month',
    'is_first_month' y 'is_last_month'.

    Parameters:
        df (DataFrame): El DataFrame con los datos resampleados.

    Returns:
        DataFrame: El DataFrame con las características agregadas.

    """
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['days_in_month'] = df.index.days_in_month
    df['is_first_month'] = df.index.to_series().apply(lambda x: is_first_or_last_month(x, is_first=True))
    df['is_last_month'] = df.index.to_series().apply(is_first_or_last_month)
    return df

def change_frequency(df, new_freq='M', plot=True):
    """
    Cambia la frecuencia del DataFrame resampleando los datos y realiza la visualización, el promedio de gastos y la
    ingeniería de características.

    Parameters:
        df (DataFrame): El DataFrame original.
        new_freq (str): La nueva frecuencia para resamplear el DataFrame. Por defecto es 'M' (mes).
        plot (bool): Indica si se debe graficar los datos resampleados. Por defecto es True.

    Returns:
        Tuple: Una tupla que contiene el DataFrame resampleado y el DataFrame con características agregadas.

    """
    resampled_data = resample_dataframe(df, new_freq)
    if plot:
        visualize_resampled_data(resampled_data)
    print('Promedio gastos', new_freq, locale.currency(resampled_data.values.mean(), grouping=True))
    df_time = resampled_data.to_frame()
    df_time = add_holidays_to_frame(resampled_data, df_time)
    df_time = feature_engineering(df_time)
    return resampled_data, df_time

def save_dataframe_as_csv(df, directory, file_name):
    """
    Save a DataFrame to a CSV file while preserving the index and date frequency.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        directory (str): The directory path where the CSV file will be saved.
        file_name (str): The name of the CSV file.

    Returns:
        None
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)
    file_path_to_save = os.path.join(directory, file_name)
    df.to_csv(file_path_to_save, index=True, date_format='%Y-%m-%d')

def load_dataframe_from_csv(file_path, index_column='date'):
    """
    Load a CSV file into a Pandas DataFrame and restore the index.

    Args:
        file_path (str): The path of the CSV file to be loaded.
        index_column (str): The name of the column representing the index.

    Returns:
        pandas.DataFrame: The loaded DataFrame with the restored index.
    """
    return pd.read_csv(file_path, index_col=index_column, parse_dates=True)

# def change_fecuency(df, new_freq= 'M'):
#     resample_date = df.resample(new_freq).cost.sum()
#     resample_date.drop(resample_date.tail(1).index,inplace=True)
#     #agragué
#     # df_time.index.freq = 'M'
#     plt.figure(figsize=(14,5))
#     # plt.plot(resample_date)
#     fig = px.line(resample_date, x=resample_date.index, y="cost", title='Gastos en el tiempo')
#     fig.show()
#     print('Promedio gastos', new_freq, locale.currency(resample_date.values.mean(), grouping=True))
#     df_time = resample_date.to_frame()
#     df_time = add_holidays_to_frame(resample_date, df_time)
#     # feature eniniering
#     # https://pandas.pydata.org/pandas-docs/version/0.23/api.html#datetimelike-properties
#     df_time['month'] = df_time.index.month
#     df_time['year'] = df_time.index.year
#     df_time['days_in_month'] = df_time.index.days_in_month

#     # saber si es diciembre un mes de altos gastos o enero tambien un mes con altos gastos
#     df_time['is_first_month'] = df_time.index.to_series().apply(lambda x: is_first_or_last_month(x, is_first=True))
#     df_time['is_last_month'] = df_time.index.to_series().apply(is_first_or_last_month)

#     return resample_date, df_time


def metrict_error(y_true, y_pred):
    print('------ Métricas dependientes de la escala  --------')
    MAE = mean_absolute_error(y_true, y_pred)
    print('Error MAE    : ',  locale.currency(MAE, grouping=True))
    print('Error MSE    : ',mean_squared_error(y_true, y_pred))
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    print('Error RMSE   : ', locale.currency(RMSE, grouping=True))

    print('------ Métricas de porcentaje de error  --------')
    print('Error MAPE    : ',round(mean_absolute_percentage_error(y_true, y_pred) *100, 1), '%')
    print(f"SMAPE        : {round(smape(y_true, y_pred),2)}")


def train_test_time_series(ts):
    # Train Test Split Index
    train_size = 0.8
    split_idx = round(len(ts)* train_size)
    split_idx

    # Split
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]

    # Visualize split
    # fig,ax= plt.subplots(figsize=(12,8))
    # kws = dict(marker='o')
    # plt.plot(train, label='Train', **kws)
    # plt.plot(test, label='Test', **kws)
    # ax.legend(bbox_to_anchor=[1,1]);
    return train, test

def split_train_test_data(data, target_variable, test_size=0.2):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        data (pd.DataFrame): El DataFrame que contiene los datos.
        target_variable (str): El nombre de la variable objetivo.
        test_size (float, optional): El tamaño deseado del conjunto de prueba. Por defecto es 0.2.

    Returns:
        tuple: Una tupla que contiene los conjuntos de entrenamiento y prueba para las variables exógenas
        y la variable objetivo.
    """
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]
    train_size = int(len(data) * (1 - test_size))
    train_X, train_y = X[:train_size], y[:train_size]
    test_X, test_y = X[train_size:], y[train_size:]
    return train_X, train_y, test_X, test_y

def select_best_exogenous_variable(metrics_dict):
    """
    Selecciona la mejor variable exógena en función del menor valor de MSE.

    Args:
        metrics_dict (dict): Un diccionario que contiene las métricas de cada combinación de variables exógenas.

    Returns:
        tuple: Una tupla que contiene la mejor variable exógena, una descripción de por qué se seleccionó
        y las métricas correspondientes a esa combinación.
    """
    min_mse = float('inf')
    best_variable = None
    best_metrics = None

    for variable, metrics in metrics_dict.items():
        mse = metrics['mse']
        if mse < min_mse:
            min_mse = mse
            best_variable = variable
            best_metrics = metrics

    description = f"La variable exógena {best_variable} tiene el menor MSE, lo que indica un mejor ajuste del modelo y es relevante para el problema."

    return best_variable, description, best_metrics




def build_evaluate_sarima_model(train_X, train_y, test_X, test_y, exog_var, order, seasonal_order):
    """
    Construye y evalúa un modelo SARIMAX con una variable exógena dada.

    Args:
        train_X (pd.DataFrame): Conjunto de entrenamiento para las variables exógenas.
        train_y (pd.Series): Conjunto de entrenamiento para la variable objetivo.
        test_X (pd.DataFrame): Conjunto de prueba para las variables exógenas.
        test_y (pd.Series): Conjunto de prueba para la variable objetivo.
        exog_var (str): La variable exógena actual.
        order (tuple): El orden del modelo SARIMA.
        seasonal_order (tuple): El orden estacional del modelo SARIMA.

    Returns:
        tuple: Una tupla que contiene el modelo ajustado, las predicciones y las métricas correspondientes.
    """
    model_sarima = sm.tsa.SARIMAX(train_y, exog=train_X[exog_var], order=order, seasonal_order=seasonal_order)
    model_sarima_fit = model_sarima.fit(disp=False)
    y_pred_sarima = model_sarima_fit.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1, exog=test_X[exog_var])
    metrics = evaluate_forecast(test_y, y_pred_sarima, False)
    return model_sarima_fit, y_pred_sarima, metrics

def plot_predict(train, test, forecast):
    train_plot = train
    train_plot['type'] = 'train'
    test_plot = test
    test_plot['type'] = 'test'
    forecast_plot = forecast
    forecast_plot['type'] = 'predict'
    forecast_plot = pd.concat([test_plot, train_plot, forecast_plot ],ignore_index=True)
    forecast_plot
    fig = px.line(forecast_plot, x='date', y="value", color="type",hover_data={"date": "|%B %d, %Y"}, title='Resultados de la predicción')
    fig.show()
    return forecast_plot

def precict_transform(model, train, test, y_col, period_test, is_diff=False, exogen_labels=[]):
    end = len(train) + len(test)
    exogx = train[exogen_labels]
    exogx_test = test[exogen_labels]
    has_exogen = bool(len(exogx) and len(exogx_test))

    if has_exogen:
        print('Aplica variables exógenas')
        forecast, conf_int = model.predict(n_periods=period_test, exogenous=exogx_test, return_conf_int=True)
    else:
        forecast, conf_int = model.predict(n_periods=period_test, return_conf_int=True)

    index_of_fc = pd.date_range(test.index[0], periods = period_test, freq='M')

    if is_diff:
        df_auto_pred = pd.DataFrame(forecast, columns=['value'])
        last_original_train = train.iloc[-1][y_col]# último dato antes diferenciar
        df_forescast = last_original_train + df_auto_pred.cumsum()
        df_forescast = df_forescast.set_index(index_of_fc)
        plot_predi = pd.DataFrame(data= {'value': df_forescast['value'], 'date': df_forescast.index})
        # d = {'lower': lower_series, 'upper': upper_series, 'pred':df_forescast[['value']]}
        # pred_df = pd.DataFrame(data=d)
    else:
        # make series for plotting purpose
        fitted_series = pd.Series(forecast, index=index_of_fc)
        lower_series = pd.Series(conf_int[:, 0], index=index_of_fc)
        upper_series = pd.Series(conf_int[:, 1], index=index_of_fc)
        d = {'lower': lower_series, 'upper': upper_series, 'pred':forecast}
        pred_df = pd.DataFrame(data=d)
        plot_predi = pd.DataFrame(data= {'value': pred_df['pred'], 'date': pred_df.index})




    plot_train = pd.DataFrame(data= {'value': train[y_col], 'date': train.index})
    plot_test = pd.DataFrame(data= {'value': test[y_col], 'date': test.index})

    metrict_error( test[y_col], plot_predi['value'] )
    forecast_plot = plot_predict(plot_train, plot_test, plot_predi)
    return forecast_plot
def precict_transform_2(model, train, test, y_col, var_exoge):
    end = len(train) + len(test)
    exogenus = test[var_exoge]
    forecast = model.predict(start=len(train), end=end-1, exog=exogenus)
    index_of_fc = pd.date_range(test.index[0], periods = len(forecast), freq='M')
    print('LEN index_of_fc:', len(index_of_fc))
    print('LEN forecast:', len(forecast))

        # make series for plotting purpose
    fitted_series = pd.Series(forecast, index=index_of_fc)
    # lower_series = pd.Series(conf_int[:, 0], index=index_of_fc)
    # upper_series = pd.Series(conf_int[:, 1], index=index_of_fc)

    d = {'pred':forecast}
    pred_df = pd.DataFrame(data=d)
    plot_train = pd.DataFrame(data= {'value': train[y_col], 'date': train.index})
    plot_test = pd.DataFrame(data= {'value': test[y_col], 'date': test.index})
    plot_predi = pd.DataFrame(data= {'value': pred_df['pred'], 'date': pred_df.index})
    print('--- ------- -------- -------------')
    print('TEST: ', test[y_col].shape)
    print('PRED: ', pred_df['pred'].shape)
    metrict_error( test[y_col], pred_df['pred'] )
    # print('Error MSE    : ',mean_squared_error(test[y_col],pred_df['pred']))
    # print('Error MAE   : ',mean_absolute_error(test[y_col],pred_df['pred']))
    forecast_plot = plot_predict(plot_train, plot_test, plot_predi)
    return forecast_plot

def evaluate_forecast(actual, predicted, print_table=True, use_markdown=False):
    """
    Evalúa las métricas de error entre los valores reales y los valores predichos de una serie temporal.

    Args:
        actual (array-like): Valores reales de la serie temporal.
        predicted (array-like): Valores predichos de la serie temporal.
        print_table (bool, optional): Indica si se debe imprimir la tabla de métricas. Por defecto es True.
        use_markdown (bool, optional): Indica si se debe utilizar el formato Markdown en la impresión de la tabla.
                                       Por defecto es False.

    Returns:
        dict: Un diccionario que contiene las métricas de error calculadas.

    """
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    RMSE = mean_squared_error(actual, predicted, squared=False)
    MAPE = mean_absolute_percentage_error(actual, predicted)
    SMAPE = smape(actual, predicted)
    MAPE_percent = round(MAPE * 100, 2)

    # Configurar locale para mostrar los valores en pesos colombianos
    locale.setlocale(locale.LC_MONETARY, 'es_CO.UTF-8')

    # Crear una tabla con las métricas
    table_data = [
        ["**Scale-Dependent Metrics**", ""],
        ["Mean Squared Error (MSE):", locale.currency(mse, grouping=True)],
        ["Mean Absolute Error (MAE):", locale.currency(mae, grouping=True)],
        ["Root Mean Squared Error (RMSE):", locale.currency(RMSE, grouping=True)],
        ["**Percentage-Error Metrics**", ""],
        ["Mean Absolute percentage Error (MAPE)", f'{MAPE_percent}%'],
        ["Symmetric Mean Absolute percentage Error (SMAPE)", f'{round(SMAPE, 2)}%']
    ]

    if print_table:
        if use_markdown:
            markdown_table = tabulate(table_data, headers=["Metric", "Value"], tablefmt="pipe")
            display(Markdown(markdown_table))
        else:
            print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # Retornar las métricas en un diccionario
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': RMSE,
        'MAPE': MAPE,
        'SMAPE': SMAPE,
        'MAPE_percent': MAPE_percent
    }

    return metrics

def load_dataset(full_path):
    """
    Carga un conjunto de datos desde un archivo CSV y realiza algunas transformaciones en el DataFrame resultante.

    Args:
        full_path (str): Ruta completa del archivo CSV a cargar.

    Returns:
        pandas.DataFrame: DataFrame cargado a partir del archivo CSV con las transformaciones realizadas.
    """
    df = pd.read_csv(full_path, parse_dates=True)
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.tz_localize(None)
    df.set_index('date', inplace=True)
    df.index.freq = 'M'
    return df

def plot_time_series(train_y, test_y, y_pred, title='Serie de Tiempo - Train, Test y Predicciones'):
    """
    Grafica los datos de entrenamiento (train), prueba (test) y las predicciones de un modelo de series de tiempo.

    Args:
        train_y (pandas.Series): Serie de tiempo de los datos de entrenamiento.
        test_y (pandas.Series): Serie de tiempo de los datos de prueba.
        y_pred (pandas.Series): Serie de tiempo de las predicciones.

    Returns:
        None (Muestra el gráfico utilizando Plotly)

    """

    # Concatenar los datos de entrenamiento, prueba y predicciones en un solo DataFrame
    df = pd.DataFrame({'Train': train_y, 'Test': test_y, 'Predictions': y_pred})

    # Crear una figura de Plotly
    fig = go.Figure()

    # Agregar la serie de tiempo de entrenamiento al gráfico
    fig.add_trace(go.Scatter(x=train_y.index, y=train_y, name='Train'))

    # Agregar la serie de tiempo de prueba al gráfico
    fig.add_trace(go.Scatter(x=test_y.index, y=test_y, name='Test'))

    # Agregar la serie de tiempo de predicciones al gráfico
    fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, name='Predictions'))

    # Personalizar el diseño del gráfico
    fig.update_layout(title=title,
                      xaxis_title='Fecha',
                      yaxis_title='Valor',
                      legend=dict(x=0, y=1),
                      height=500)

    # Mostrar el gráfico
    fig.show()

def time_series_interpolation(time_series, anomaly_index):
    """
    Performs interpolation on a time series by imputing missing values at the specified anomaly index.

    Parameters:
        time_series (pd.Series): The original time series.
        anomaly_index: The index of the anomaly where the missing value occurs.

    Returns:
        pd.Series: The time series with missing values imputed using linear interpolation.
    """
    # Create a copy of the original time series
    interpolated_series = time_series.copy()

    # Get the anomalous value
    anomaly_value = interpolated_series.loc[anomaly_index]

    # Replace the anomalous value with NaN (missing values)
    interpolated_series.loc[anomaly_index] = np.nan

    # Perform linear interpolation to impute the missing values
    interpolated_series.interpolate(method='linear', inplace=True)

    return interpolated_series

def detect_anomalies(data, column, threshold_factor=2):
    """
    Detects anomalies in a specified column of a DataFrame using mean and standard deviation approach.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name where anomalies should be detected.
        threshold_factor (float, optional): The factor to multiply the standard deviation to determine the threshold.

    Returns:
        pd.DataFrame: A DataFrame containing the rows with detected anomalies.
    """
    mean = data[column].mean()
    std = data[column].std()
    threshold = threshold_factor * std
    anomalies = data[data[column].abs() - mean > threshold]

    return anomalies

def fit_auto_arima(train_y, train_X):
    # Ajustar modelo Auto ARIMA
    model = pm.auto_arima(train_y, exogenous=train_X, seasonal=True)
    return model

def fit_sarima(train_y, train_X, order, seasonal_order):
    # Ajustar modelo SARIMA
    model = SARIMAX(train_y, exog=train_X, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

def save_results_to_file(model_metrics, file_path):
    # Guardar los resultados en un archivo CSV
    result_df = pd.DataFrame(model_metrics)
    result_df.to_csv(file_path, index=False)