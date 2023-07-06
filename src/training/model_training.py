class ModelTraining:

    def __init__(self, utils):
      self.utils = utils
      self.data = None
      self.data_without_outliers = None
      self.datasets = None

    def preprocess_data(self):
      file_name = "./data/processed/df_time_monthly.csv"
      file_name_without_outliers = "./data/processed/df_time_monthly_without_outliers.csv"
      self.data = self.utils.load_from_csv(file_name)
      self.data_without_outliers = self.utils.load_from_csv(file_name_without_outliers)
      X = self.data.drop('cost', axis=1)
      y = self.data['cost']
      X_without_outliers = self.data_without_outliers.drop('cost', axis=1)
      y_without_outliers = self.data_without_outliers['cost']
      self.datasets = [
          {'name': 'Normal', 'train_ratio': 0.8, 'X': X, 'y': y},
          {'name': 'Without Outliers', 'train_ratio': 0.8, 'X': X_without_outliers, 'y': y_without_outliers}
          # Add more datasets if needed
      ]

    def train_and_evaluate(self):
      for dataset in self.datasets:
          dataset_name = dataset['name']
          X = dataset['X'].values
          y = dataset['y'].values
          for train_index, test_index in self.tscv.split(X):
              train_X, test_X = X[train_index], X[test_index]
              train_y, test_y = y[train_index], y[test_index]
              test_X_df = pd.DataFrame(test_X, index=pd.to_datetime(dataset['X'].index[test_index]))

              model_arima_fit = self.fit_arima(train_y, train_X)
              model_sarima_fit = self.fit_sarima(train_y, train_X, model_arima_fit.order, model_arima_fit.seasonal_order)

              for model_name, model in self.models:
                  result, y_pred = self.evaluate_model(model_name, model, train_X, train_y, test_X, test_y)
                  self.model_metrics.append(result)
                  y_pred_series = pd.Series(y_pred, index=test_X_df.index)
                  plot_time_series(pd.Series(train_y, index=pd.to_datetime(dataset['X'].index[train_index])),
                                    pd.Series(test_y, index=pd.to_datetime(dataset['X'].index[test_index])),
                                    y_pred_series,
                                    f'Serie Tiempo - {dataset_name} - {model_name}')
                  print('---')

              result_arima, y_pred_arima = self.evaluate_arima(model_arima_fit, train_y, test_y, test_X_df, dataset_name)
              self.model_metrics.append(result_arima)
              y_pred_arima_series = pd.Series(y_pred_arima, index=test_X_df.index)
              plot_time_series(pd.Series(train_y, index=pd.to_datetime(dataset['X'].index[train_index])),
                                pd.Series(test_y, index=pd.to_datetime(dataset['X'].index[test_index])),
                                y_pred_arima_series,
                                f'Serie Tiempo - {dataset_name} - Auto ARIMA')
              print('---')

              result_sarima, y_pred_sarima = self.evaluate_sarima(model_sarima_fit, train_y, test_y, test_X_df, dataset_name)
              self.model_metrics.append(result_sarima)
              y_pred_sarima_series = pd.Series(y_pred_sarima, index=test_X_df.index)
              plot_time_series(pd.Series(train_y, index=pd.to_datetime(dataset['X'].index[train_index])),
                                pd.Series(test_y, index=pd.to_datetime(dataset['X'].index[test_index])),
                                y_pred_sarima_series,
                                f'Serie Tiempo - {dataset_name} - SARIMA')
              print('---')