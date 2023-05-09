import pandas as pd
import locale
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pmdarima.metrics import smape
# handle time
import locale
locale.setlocale(locale.LC_MONETARY, 'es_CO.UTF-8')
import joblib
class Utils:
    def load_from_csv(self, path):
        """
        Load data from a CSV file.

        Parameters
        ----------
        path : str
            The path to the CSV file.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the data from the CSV file.
        """
        df= pd.read_csv(path, parse_dates=True)
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.tz_localize(None)
        df.set_index('date', inplace=True)
        df.index.freq = 'M'
        return df

    def model_export(self,  cls, score):
        score_str = str(score).replace(".", "_")
        file_name = f'best_model_{score_str}.pkl'
        joblib.dump(cls, f'./models/{file_name}')

    def load_model(self, path):
        return joblib.load(path)

    def train_test_time_series(self, ts):
      """
      Split a time series into training and testing sets.

      Parameters
      ----------
      ts : pandas.Series
          The time series data to split.

      Returns
      -------
      tuple
          A tuple containing two pandas.Series objects, the training set
          and the testing set.
      """
      # Train Test Split Index
      train_size = 0.8
      split_idx = round(len(ts)* train_size)
      split_idx
      # Split
      train = ts.iloc[:split_idx]
      test = ts.iloc[split_idx:]
      return train, test

    def metric_error(self, y_true, y_pred):
        """
        Computes and prints various error metrics between predicted and true values.

        Parameters:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

        Raises:
        ValueError: If y_true and y_pred are not of the same length.
        Returns:
        - None.

        Imprime en consola diferentes métricas de evaluación:
        - Error MAE.
        - Error MSE.
        - Error RMSE.
        - Error MAPE.
        - Error SMAPE.

        Ejemplo:
        >>> y_true = [1, 2, 3, 4, 5]
        >>> y_pred = [1, 3, 3, 4, 4]
        >>> metric_error(y_true, y_pred)
        ------ Métricas dependientes de la escala  --------
        Error MAE    :  $0.80
        Error MSE    :  0.6
        Error RMSE   :  $0.78
        ------ Métricas de porcentaje de error  --------
        Error MAPE    :  22.7 %
        SMAPE        : 11.25

        """
        MAE = mean_absolute_error(y_true, y_pred)
        MSE = mean_squared_error(y_true, y_pred)
        RMSE = mean_squared_error(y_true, y_pred, squared=False)
        MAPE = mean_absolute_percentage_error(y_true, y_pred)
        SMAPE = smape(y_true, y_pred)
        MAPE_percent = round(MAPE * 100, 1)
        results = [('Error MAE', locale.currency(MAE, grouping=True)),
                ('Error MSE', MSE),
                ('Error RMSE', locale.currency(RMSE, grouping=True)),
                ('Error MAPE', f'{MAPE_percent}%'),
                ('SMAPE', round(SMAPE, 2))]
        return results
