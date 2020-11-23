import datetime
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


class SalesPredictor:
    def __init__(self, period, residuals_model):
        self.period = period
        self.residuals_model = residuals_model
        self.seasonal = None
        self.trend = None
        self.first_week = 0
        
    def _get_week_number(self, full_date):
        date_parts = [int(el) for el in full_date.split('-')]
        a_date = datetime.date(date_parts[0], date_parts[1], date_parts[2])
        return a_date.isocalendar()[1]
    
    def fit(self, prices, sales):
        # TODO: implement a training procedure. "prices" has a datetime index.
        # The method should: 1. extract & save seasonal component; 2. train estimator on residuals.
        # Note that a first week number should be saved in order to determine seasonality in the future.
        decomposition = seasonal_decompose(x=sales, model='additive', period=self.period, extrapolate_trend='freq')
        self.seasonal = (decomposition.seasonal + decomposition.trend)[:self.period]

        self.first_week = self._get_week_number(prices.index[0])
        self.residuals_model.fit(prices, decomposition.resid)
    
    def _predict_array(self, week, prices):
        # TODO: Implement prediction procedure. Use extracted seasonality and model trained on residuals.
        particular_seasonal_decompose = []
        if isinstance(week, np.ndarray) and len(week) != 1:
            for week_date in week:
                week_num = self._get_week_number(week_date)
                seasonal_index = week_num - self.first_week if week_num >= self.first_week else self.period + 1 - self.first_week
                particular_seasonal_decompose.append(self.seasonal[seasonal_index])
        else: # week is [int]
            week_num = week[0]
            seasonal_index = week_num - self.first_week if week_num >= self.first_week else self.period + 1 - self.first_week
            particular_seasonal_decompose.append(self.seasonal[seasonal_index])
        
        return self.residuals_model.predict(prices) + particular_seasonal_decompose

    def predict(self, week, prices):
        if not isinstance(week, np.ndarray) and not isinstance(prices, np.ndarray):
            sales_pred = self._predict_array(np.array([week]), np.array([[prices]]))
            return sales_pred[0]
        return self._predict_array(week, prices)
    
    def __repr__(self):
        return "SalesPredictor() with " + str(self.residuals_model)
