import pandas as pd
import holidays
import datetime
#handle error
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima.metrics import smape
# handle time
import locale
locale.setlocale(locale.LC_MONETARY, 'es_CO.UTF-8')

co_holidays = holidays.CO()
import plotly.express as px
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
    month = date.month
    return month == 1 if is_first else month == 12


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