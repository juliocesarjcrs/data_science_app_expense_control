import pandas as pd
from utils import Utils
from time_series import TimeSeries
if __name__ == "__main__":
    utils = Utils()
    # # case 0 generate folder strucutre
    # folder_path = "./"
    # exclude_dirs = [".git", "myenv-py3.11", "__pycache__"]
    # folder_structure = utils.generate_folder_structure(folder_path, exclude_dirs)

    # case 2 API training model and save new model
    # data = utils.load_from_csv('./in/preprocess/df_time_monthly')
    # print(data.head())

    # timeSeries = TimeSeries()
    # train_data, test_data = utils.train_test_time_series(data)
    # print(train_data.shape[0])
    # print(test_data.shape[0])
    # my_model = timeSeries.training_model('SARIMAX', train_data)
    # params = {'train': train_data, 'test': test_data, 'var_exoge': ['days_in_month']}
    # my_predict = timeSeries.predict_model(my_model, params)
    # results = utils.metric_error(test_data['cost'].values, my_predict.values)
    # for result in results:
    #     print(f'{result[0]}: {result[1]}')
    # utils.model_export(my_model, results[3][1])

    # case 2 API load model and predict

    best_model = utils.load_model('./models/best_model_47_5%.pkl')

    # Generación de fechas para los próximos 5 periodos
    future_dates = pd.date_range(start= '2023-05-01', periods=5, freq='M') # M month end frequency
    exogen_data = []
    for date in future_dates:
        exogen_data.append(date.days_in_month)
    print(exogen_data)
    print(future_dates)

    # Predicción de valores futuros
    my_exog_data = [31, 30, 31, 31, 30]
    future_predictions = best_model.forecast(steps=5, exog=my_exog_data)
    print(future_predictions)
