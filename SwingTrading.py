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
    
    def add_moving_averages(self):
        """
        Adds the moving average columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the moving average columns
        """
        self.data['MA_5'] = self.data['Close'].rolling(5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(10).mean()
        self.data['MA_20'] = self.data['Close'].rolling(20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(50).mean()
        self.data['MA_100'] = self.data['Close'].rolling(100).mean()
        self.data['MA_200'] = self.data['Close'].rolling(200).mean()

        return self.data

    def add_bollinger_bands(self):
        """
        Adds the Bollinger Bands columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Bollinger Bands columns
        """
        self.data['Mid_Bollinger_Band'] = self.data['Close'].rolling(20).mean()
        self.data['Upper_Bollinger_Band'] = self.data['Mid_Bollinger_Band'] + 2 * self.data['Close'].rolling(20).std() 
        self.data['Lower_Bollinger_Band'] = self.data['Mid_Bollinger_Band'] - 2 * self.data['Close'].rolling(20).std()

        return self.data

    def add_macd(self):
        """
        Adds the MACD columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the MACD columns
        """
        self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26).mean()
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9).mean()

        return self.data
        
    def add_rsi(self):
        """
        Adds the Relative Strength Index column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Relative Strength Index column
        """
        self.data['Change'] = self.data['Close'] - self.data['Close'].shift(1)
        self.data['Gain'] = self.data['Change'].apply(lambda x: x if x > 0 else 0)
        self.data['Loss'] = self.data['Change'].apply(lambda x: -x if x < 0 else 0)
        self.data['Avg_Gain_14'] = self.data['Gain'].rolling(14).mean()
        self.data['Avg_Loss_14'] = self.data['Loss'].rolling(14).mean()
        self.data['RS'] = self.data['Avg_Gain_14'] / self.data['Avg_Loss_14']
        self.data['RSI'] = 100 - (100 / (1 + self.data['RS']))

        return self.data
        
        
    def add_vwap(self):
        """
        Adds the VWAP column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the VWAP column
        """
        self.data['VWAP'] = (self.data['Close'] * self.data['Volume']).cumsum() / self.data['Volume'].fillna(0).cumsum()

        return self.data
        
    def add_volatility(self):
        """
        Adds the volatility columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the volatility columns
        """
        self.data['Vol_5'] = self.data['Close'].rolling(5).std()
        self.data['Vol_10'] = self.data['Close'].rolling(10).std()
        self.data['Vol_20'] = self.data['Close'].rolling(20).std()
        self.data['Vol_50'] = self.data['Close'].rolling(50).std()
        self.data['Vol_100'] = self.data['Close'].rolling(100).std()
        self.data['Vol_200'] = self.data['Close'].rolling(200).std()

        return self.data
        
    def add_accumulation_distribution(self):
        """
        Adds the accumulation/distribution column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the accumulation/distribution column
        """
        self.data['Money_Flow_Multiplier'] = (2*self.data['Close'] - self.data['High'] - self.data['Low'])/(self.data['High'] - self.data['Low'])
        self.data['Money_Flow_Volume'] = self.data['Money_Flow_Multiplier'] * self.data['Volume']
        self.data['ADL'] = self.data['Money_Flow_Volume'].cumsum()

        return self.data
        
    def add_on_balance_volume(self):
        """
        Adds the on-balance volume column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the on-balance volume column
        """
        self.data['OBV'] = np.where(self.data.Close.diff(1) > 0, self.data.Volume, 
                                    np.where(self.data.Close.diff(1) < 0, -self.data.Volume, 0)).cumsum()

        return self.data
        
    def add_stochastic_oscillator(self):
        """
        Adds the stochastic oscillator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the stochastic oscillator columns
        """
        self.data['Stochastic_Oscillator_K'] = 100 * ((self.data['Close'] - self.data['Low'].rolling(14).min()) / (self.data['High'].rolling(14).max() - self.data['Low'].rolling(14).min()))
        self.data['Stochastic_Oscillator_D'] = self.data['Stochastic_Oscillator_K'].rolling(3).mean()

        return self.data
        
    def add_williams(self):
        """
        Adds the Williams %R columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Williams %R columns
        """
        self.data['Williams_R'] = -100 * ((self.data['High'].rolling(14).max() - self.data['Close']) / (self.data['High'].rolling(14).max() - self.data['Low'].rolling(14).min()))

        return self.data
    
    def add_cci(self):
        """
        Adds the Commodity Channel Index column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Commodity Channel Index column
        """
        self.data['TP'] = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        self.data['CCI'] = (self.data['TP'] - self.data['TP'].rolling(20).mean()) / (0.015 * self.data['TP'].rolling(20).std())

        return self.data
    
    def add_fibonacci(self):
        """
        Adds the Fibonacci columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Fibonacci columns
        """
        self.data['Fibonacci_0.236'] = self.data['Close'].shift(-1) * 0.236
        self.data['Fibonacci_0.382'] = self.data['Close'].shift(-1) * 0.382
        self.data['Fibonacci_0.50'] = self.data['Close'].shift(-1) * 0.50
        self.data['Fibonacci_0.618'] = self.data['Close'].shift(-1) * 0.618
        self.data['Fibonacci_1.00'] = self.data['Close'].shift(-1) * 1.00
        self.data['Fibonacci_1.27'] = self.data['Close'].shift(-1) * 1.27
        self.data['Fibonacci_1.618'] = self.data['Close'].shift(-1) * 1.618

        return self.data

    def add_label(self):
        """
        Adds the label column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the label column
        """
        self.data['Label'] = np.where(self.data['Close'] > self.data['Close'].shift(-1), 
            'Positive', np.where(self.data['Close'] < self.data['Close'].shift(-1), 'Negative', 'Neutral'))

        return self.data
    
    def add_momentum(self):
        """
        Adds the momentum column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the momentum column
        """
        self.data['Momentum_10'] = self.data['Close'].diff(10)

        return self.data
    
    def add_aroon(self):
        """
        Adds the Aroon columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Aroon columns
        """
        self.data['Aroon_Up'] = 100 * (self.data['High'].rolling(25).apply(lambda x: list(x).index(max(x))+1) / 25)
        self.data['Aroon_Down'] = 100 * (self.data['Low'].rolling(25).apply(lambda x: list(x).index(min(x))+1) / 25)

        return self.data
    
    def add_chaikin_oscillator(self):
        """
        Adds the Chaikin Oscillator column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Chaikin Oscillator column
        """
        self.data['ADL_10'] = self.data['ADL'].rolling(10).mean()
        self.data['Chaikin_Osc'] = self.data['ADL_10'] - self.data['ADL']

        return self.data
    
    def add_trend_line(self):
        """
        Adds the trend line column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the trend line column
        """
        self.data['Trend_Line'] = self.data['Close'].rolling(20).mean()

        return self.data
    



# -Add Volume Profile Analysis
# -Add Candlestick Pattern Analysis
# -Add Average True Range
# -Add Momentum Indicators
# -Add Relative Strength Index Divergence
# -Add Parabolic SAR
# -Add Bollinger Band Width
# -Add Average Directional Movement Index
# -Add Ichimoku Kinko Hyo
# -Add Elliott Wave Analysis
# -Add Moving Average Convergence Divergence
# -Add On-Balance Volume
# -Add StochRSI
# -Add Keltner Channels
# -Add Gann Fan
# -Add Triple Exponential Average
# -Add Rate of Change
# -Add Detrended Price Oscillator
# -Add Fisher Transform
# -Add Mass Index
# -Add Vortex Indicator
# -Add KST Oscillator
# -Add Chaikin Money Flow
# -Add Average Directional Index
# -Add Ultimate Oscillator
# -Add Linear Regression Indicator
# -Add Negative Volume Index
# -Add McClellan Oscillator
# -Add Kaufman Adaptive Moving Average
# -Add Moving Average of Oscillator
# -Add Klinger Volume Oscillator
# -Add Coppock Curve
# -Add Elder Ray Index
# -Add Force Index
# -Add Elder Impulse System
# -Add Chandelier Exit
# -Add Keltner Channel Breakout System
# -Add Parabolic SAR and ADX Combo
# -Add DMI Stochastic System
# -Add Swing Index
# -Add Andrew’s Pitchfork
# -Add Ichimoku Cloud
# -Add Relative Strength Index Wilder Smoothing
# -Add Rainbow Charts
# -Add Market Profile
# -Add Elder Safe Zone Strategy
# -Add Moving Average Envelope
# -Add Bollinger Band Squeeze
# -Add Average True Range
# -Add Obvios Momentum Indicator
# -Add Keltner Channel Breakout System
# -Add Money Flow Index
# -Add Donchian Channel Breakout System
# -Add Average Directional Movement Index
# -Add Commodity Channel Index
# -Add Stochastic RSI
# -Add Volume Weighted Average Price
# -Add Connors RSI
# -Add SuperTrend Indicator
# -Add Average True Range Stop
# -Add Swing Index
# -Add Ichimoku Kinko Hyo
# -Add Keltner Channel Volatility Breakout System
# -Add Triple Screen Trading System
# -Add Standard Deviation Channel
# -Add Adaptive Moving Average
# -Add Volatility Stop
# -Add Volume Zone Oscillator
# -Add Volatility Adjusted Moving Average
# -Add Acceleration/Deceleration Oscillator
# -Add Directional Movement Index
# -Add Elder-Ray Bull/Bear Power
# -Add Detrended Price Oscillator
# -Add TRIX Indicator
