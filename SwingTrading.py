import yfinance as yf
import numpy as np

class SwingTrading:
    """
    A class for Swing Trading using Python
    """
    
    def __init__(self, ticker, start_date, end_date, period, max_increase, min_decrease, number_of_days):
        """
        Initialize a SwingTrading object

        Parameters
        ----------
        ticker : string
            The ticker name of the stock
        start_date : string
            The start date in the YYYY-MM-DD format
        end_date : string
            The end date in the YYYY-MM-DD format
        period : int
            The period of the data (daily, weekly, monthly)
        max_increase : float
            The max percentage of increase
        min_decrease : float
            The min percentage of decrease
        number_of_days : int
            The number of days to check for target
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.max_increase = max_increase
        self.min_decrease = min_decrease
        self.number_of_days = number_of_days
    
    def get_data(self):
        """
        Retrieves the data from yahoo finance

        Parameters
        ----------
        None
        
        Returns
        -------
        DataFrame
            A pandas DataFrame containing the stock data
        """
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, period=self.period)
        self.data.reset_index(inplace=True)
        return self.data
        
    def add_rolling_columns(self, rows_forward):
        """
        Adds rolling min or max and lag columns to the DataFrame

        Parameters
        ----------
        rows_forward : int
            The number of rows to look forward

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the rolling min or max and lag columns
        """
        column_name_max = 'High'
        rolling_max = list(self.data[column_name_max].rolling(rows_forward).max())[rows_forward:]
        max_lag = list(self.data[column_name_max].rolling(rows_forward).apply(lambda x: list(x).index(max(x))+1))[rows_forward:]
        empty_ = [0]*rows_forward
        rolling_max = rolling_max + empty_
        max_lag = max_lag + empty_
        self.data['{}_{}'.format(column_name_max, rows_forward)] = rolling_max
        max_lag_var = '{}_lag_{}'.format(column_name_max, rows_forward)
        self.data[max_lag_var] = max_lag

        column_name_min = 'Low'
        rolling_min = list(self.data[column_name_min].rolling(rows_forward).min())[rows_forward:]
        min_lag = list(self.data[column_name_min].rolling(rows_forward).apply(lambda x: list(x).index(min(x))+1))[rows_forward:]
        empty_ = [0]*rows_forward
        rolling_min = rolling_min + empty_
        min_lag = min_lag + empty_
        self.data['{}_{}'.format(column_name_min, rows_forward)] = rolling_min
        min_lag_var = '{}_lag_{}'.format(column_name_min, rows_forward)
        self.data[min_lag_var] = min_lag

        self.data['High_Low_lag_5'] = np.where(self.data[max_lag_var] > self.data[min_lag_var], 'UpTrend', 
                                            np.where(self.data[min_lag_var] > self.data[max_lag_var], 'DownTrend','NoTrend'))

        return self.data
