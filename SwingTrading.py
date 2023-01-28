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
        
    def forward_calc(self,  column, calc_type):

        # check if valid calc type
        if calc_type not in ['min', 'max']:
            raise ValueError('calc_type must be one of min or max')

        # get length of df
        df_len = len(self.data)
        
        # set up list to store results
        results = []
        
        # loop through each row
        for i in range(df_len):
            # get the index of the nth row
            row_index = i + self.number_of_days
            
            # if nth row is not in df
            if row_index >= df_len:
                # store None in results
                results.append(None)
            else:
                # get column values from nth row
                col_vals = self.data.loc[i:row_index][column]
                
                # use appropriate calculation
                if calc_type == 'min':
                    res = np.min(col_vals)
                else:
                    res = np.max(col_vals)
                # store result in results
                results.append(res)
                
        var_ = column + "_" + calc_type
        self.data[var_] = results
        
        # return list of results
        return self.data
