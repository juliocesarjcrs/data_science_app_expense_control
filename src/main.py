from dependency_inyections.container import Container
def main():
    print('paso 1')
    # Crea una instancia del contenedor y resuelve las dependencias
    container = Container()
    model_training = container.model_training()
    model_training.preprocess_data()
    # Paso 1: Preparación de datos
    # prepare_data()

    # Paso 2: Entrenamiento de modelos
    # train_models()

    # Paso 3: Selección del mejor modelo
    # best_model = select_best_model()

    # Paso 4: Otras tareas o flujos de trabajo

if __name__ == "__main__":
    main()

# import pandas as pd
# from utils.utils import Utils
# from time_series import TimeSeries
# import os.path
# if __name__ == "__main__":
#     utils = Utils()
#     # case 0 generate folder strucutre
#     folder_path = "./"
#     exclude_dirs = [".git", "myenv-py3.11", "__pycache__"]
#     folder_structure = utils.generate_folder_structure(folder_path, exclude_dirs)

    # # case 2 API training model and save new model
    # filepath = "./data/processed/df_time_monthly.csv"
    # if not os.path.isfile(filepath):
    #     raise Exception(f"No se encontró el archivo {filepath}.")
    # data = utils.load_from_csv(filepath)
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

    # # case 3 API load model and predict
    # model_path = './models/best_model_38_7%.pkl'
    # if not os.path.isfile(model_path):
    #     raise Exception(f"No se encontró el archivo {model_path}.")
    # # best_model = utils.load_model('./models/best_model_47_5%.pkl')
    # best_model = utils.load_model(model_path)

    # # Generación de fechas para los próximos 5 periodos
    # future_dates = pd.date_range(start= '2023-05-01', periods=5, freq='M') # M month end frequency
    # exogen_data = []
    # for date in future_dates:
    #     exogen_data.append(date.days_in_month)
    # print(exogen_data)
    # print(future_dates)

    # # Predicción de valores futuros
    # my_exog_data = [31, 30, 31, 31, 30]
    # future_predictions = best_model.forecast(steps=5, exog=my_exog_data)
    # print(future_predictions)




