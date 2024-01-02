from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
class ModelTraining:

    def __init__(self, utils):
      self.utils = utils
      self.data = None
      self.data_without_outliers = None
      self.datasets = []
      self.model_metrics = []
      self.tscv = TimeSeriesSplit(n_splits=5, test_size=6)

      model_order = (0,1,1) # reemplazar con los valores obtenidos en el punto 2.1.2 y 2.1.3 p, d q

      # ajustar el modelo
      model_arima = ARIMA(df['cost'], order=model_order)

      self.models = [
        # ('Linear Regression', LinearRegression()),
        # ('Random Forest', RandomForestRegressor())
        ('Arima', model_arima)
        ]

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
      from sklearn.model_selection import TimeSeriesSplit
      from sklearn.metrics import mean_squared_error
      import statsmodels.api as sm
      tscv = TimeSeriesSplit(n_splits = 4)
      rmse = []
      for train_index, test_index in tscv.split(cross_validation):
          cv_train, cv_test = cross_validation.iloc[train_index], cross_validation.iloc[test_index]
          
          arma = sm.tsa.ARMA(cv_train, (2,2)).fit(disp=False)
          
          predictions = arma.predict(cv_test.index.values[0], cv_test.index.values[-1])
          true_values = cv_test.values
          rmse.append(sqrt(mean_squared_error(true_values, predictions)))
    
print("RMSE: {}".format(np.mean(rmse)))

    def train_and_evaluate(self, use_cross_validation):
      for dataset in self.datasets:
          dataset_name = dataset['name']
          X = dataset['X'].values
          y = dataset['y'].values
          if use_cross_validation:
            self.apply_cross_validations(dataset_name, X, y)
          else:
              # Código sin el bucle anidado
              pass

    def apply_cross_validations(self, dataset_name, X, y):
      for train_index, test_index in self.tscv.split(X):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]

        for model_name, model in self.models:
            result, y_pred = self.evaluate_model(model_name, model, train_X, train_y, test_X, test_y)
            self.model_metrics.append(result)
            y_pred_series = pd.Series(y_pred, index=test_X_df.index)
            plot_time_series(pd.Series(train_y, index=pd.to_datetime(dataset['X'].index[train_index])),
                              pd.Series(test_y, index=pd.to_datetime(dataset['X'].index[test_index])),
                              y_pred_series,
                              f'Serie Tiempo - {dataset_name} - {model_name}')

    def evaluate_model(self, model_name, model, train_X, train_y, test_X, test_y):
      model.fit(train_X, train_y)
      y_pred = model.predict(test_X)
      metrics = evaluate_forecast(test_y, y_pred)
      return {
          'model_name': model_name,
          'model_fit': model,
          'metrics': metrics
      }, y_pred