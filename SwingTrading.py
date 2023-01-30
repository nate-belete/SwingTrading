import yfinance as yf
import pandas as pd
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
        empty_ = [np.nan]*rows_forward
        rolling_max = rolling_max + empty_
        max_lag = max_lag + empty_
        self.data['{}_{}'.format(column_name_max, rows_forward)] = rolling_max
        max_lag_var = '{}_forward_{}'.format(column_name_max, rows_forward)
        self.data[max_lag_var] = max_lag

        column_name_min = 'Low'
        rolling_min = list(self.data[column_name_min].rolling(rows_forward).min())[rows_forward:]
        min_lag = list(self.data[column_name_min].rolling(rows_forward).apply(lambda x: list(x).index(min(x))+1))[rows_forward:]
        empty_ = [np.nan]*rows_forward
        rolling_min = rolling_min + empty_
        min_lag = min_lag + empty_
        self.data['{}_{}'.format(column_name_min, rows_forward)] = rolling_min
        min_lag_var = '{}_forward_{}'.format(column_name_min, rows_forward)
        self.data[min_lag_var] = min_lag

        rows_forward_var = 'High_Low_forward_{}'.format(rows_forward)
        self.data[rows_forward_var] = np.where(self.data[max_lag_var] > self.data[min_lag_var], 'UpTrend', 
                                            np.where(self.data[min_lag_var] > self.data[max_lag_var], 'DownTrend','NoTrend'))

        return self.data
    
    def scale_column(self, col_name, period):
        """
        Scales a given column of a pandas DataFrame between 0 and 1 using the 
        columns prior 20 high and low values.

        Parameters
        ----------
        data : DataFrame
            The pandas DataFrame containing the column to be scaled
        col_name : string
            The name of the column to be scaled

        Returns
        -------
        DataFrame
            A pandas DataFrame with the scaled column
        """
        
        high = self.data[col_name].rolling(period).max().shift(1)
        low = self.data[col_name].rolling(period).min().shift(1)
        self.data['scaled_' + str(period) + "_" + col_name] = (self.data[col_name] - low)/(high - low)
        del self.data[col_name]
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


    def add_Accumulation_Distribution(self):
        """
        Adds the Accumulation/Distribution column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Accumulation/Distribution column
        """
        self.data['Accumulation_Distribution'] = (2 * self.data['Close'] - self.data['Low'] - self.data['High']) / (self.data['High'] - self.data['Low']) * self.data['Volume']

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
        Adds the on-balance volume column to the DataFrame using a for loop

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the on-balance volume column
        """
        self.data['OBV'] = 0
        self.data.loc[0,'OBV'] = self.data['Volume'][0]

        for i in range(1, len(self.data)):
            if self.data['Close'][i] > self.data['Close'][i-1]:
                self.data.loc[i,'OBV'] = self.data['OBV'][i-1] + self.data['Volume'][i]
            elif self.data['Close'][i] < self.data['Close'][i-1]:
                self.data.loc[i,'OBV'] = self.data['OBV'][i-1] - self.data['Volume'][i]
            else:
                self.data.loc[i,'OBV'] = self.data['OBV'][i-1]

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
        self.data['Fibonacci_0.236'] = self.data['Close'].shift(0) * 0.236
        self.data['Fibonacci_0.382'] = self.data['Close'].shift(0) * 0.382
        self.data['Fibonacci_0.50'] = self.data['Close'].shift(0) * 0.50
        self.data['Fibonacci_0.618'] = self.data['Close'].shift(0) * 0.618
        self.data['Fibonacci_1.00'] = self.data['Close'].shift(0) * 1.00
        self.data['Fibonacci_1.27'] = self.data['Close'].shift(0) * 1.27
        self.data['Fibonacci_1.618'] = self.data['Close'].shift(0) * 1.618

        return self.data

    def add_label(self):
        """
        Adds the label column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the label column
        """
        self.data['Label'] = np.where(self.data['Close'] > self.data['Close'].shift(1), 
            'Positive', np.where(self.data['Close'] < self.data['Close'].shift(1), 'Negative', 'Neutral'))

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
    
    def add_Volume_Profile_Analysis(self):
        """
        Adds the Volume Profile Analysis column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Volume Profile Analysis column
        """
        self.data['Volume_Profile_Analysis'] = self.data['Volume'] / self.data['Close']

        return self.data

    def add_Candlestick_Pattern_Analysis(self):
        """
        Adds the Candlestick Pattern Analysis columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Candlestick Pattern Analysis columns
        """
        self.data['Bullish_Engulfing'] = 0
        self.data['Bearish_Engulfing'] = 0
        self.data['Hammer'] = 0
        self.data['Shooting_Star'] = 0
        self.data['Doji'] = 0
        self.data['Harami'] = 0
        self.data['Hanging_Man'] = 0
        self.data['Inverted_Hammer'] = 0
        self.data['Tweezer_Bottom'] = 0
        self.data['Tweezer_Top'] = 0
        self.data['Bullish_Three_Line_Strike'] = 0
        self.data['Bearish_Three_Line_Strike'] = 0
        self.data['Bullish_Abandoned_Baby'] = 0
        self.data['Bearish_Abandoned_Baby'] = 0
        self.data['Bullish_Kicker'] = 0
        self.data['Bearish_Kicker'] = 0
        self.data['Bullish_Kick_Backs'] = 0
        self.data['Bearish_Kick_Backs'] = 0
        self.data['Bullish_Harami_Cross'] = 0
        self.data['Bearish_Harami_Cross'] = 0
        self.data['Bullish_Engulfing_Pattern'] = 0
        self.data['Bearish_Engulfing_Pattern'] = 0

        for i in range(1, len(self.data)):
            if self.data['Open'][i] > self.data['Close'][i-1] and self.data['Close'][i] > self.data['Open'][i-1]:
                self.data.loc[i,'Bullish_Engulfing'] = 1
            elif self.data['Open'][i] < self.data['Close'][i-1] and self.data['Close'][i] < self.data['Open'][i-1]:
                self.data.loc[i,'Bearish_Engulfing'] = 1
            elif self.data['Low'][i] == self.data['Open'][i] and self.data['Low'][i] < self.data['Close'][i] and (self.data['Close'][i] - self.data['Low'][i]) > (self.data['High'][i] - self.data['Close'][i]):
                self.data.loc[i,'Hammer'] = 1
            elif self.data['High'][i] == self.data['Open'][i] and self.data['High'][i] > self.data['Close'][i] and (self.data['High'][i] - self.data['Open'][i]) > (self.data['Close'][i] - self.data['Low'][i]):
                self.data.loc[i,'Shooting_Star'] = 1
            elif self.data['Open'][i] == self.data['Close'][i]:
                self.data.loc[i,'Doji'] = 1
            elif self.data['Open'][i] > self.data['Close'][i-1] and self.data['Open'][i] > self.data['Close'][i] and self.data['Close'][i-1] > self.data['Open'][i]:
                self.data.loc[i,'Harami'] = 1
            elif self.data['High'][i] == self.data['Open'][i] and self.data['High'][i] > self.data['Close'][i] and (self.data['High'][i] - self.data['Open'][i]) < (self.data['Close'][i] - self.data['Low'][i]):
                self.data.loc[i,'Hanging_Man'] = 1
            elif self.data['Low'][i] == self.data['Open'][i] and self.data['Low'][i] < self.data['Close'][i] and (self.data['Close'][i] - self.data['Low'][i]) < (self.data['High'][i] - self.data['Close'][i]):
                self.data.loc[i,'Inverted_Hammer'] = 1
            elif self.data['Low'][i] > self.data['Low'][i-1] and self.data['High'][i] < self.data['High'][i-1] and self.data['Open'][i] > self.data['Close'][i-1]:
                self.data.loc[i,'Tweezer_Bottom'] = 1
            elif self.data['High'][i] > self.data['High'][i-1] and self.data['Low'][i] < self.data['Low'][i-1] and self.data['Open'][i] < self.data['Close'][i-1]:
                self.data.loc[i,'Tweezer_Top'] = 1
            elif self.data['Close'][i] > self.data['Open'][i] and self.data['Close'][i] > self.data['Close'][i-1] and self.data['Close'][i-1] > self.data['Close'][i-2] and self.data['Close'][i-2] > self.data['Close'][i-3]:
                self.data.loc[i,'Bullish_Three_Line_Strike'] = 1
            elif self.data['Close'][i] < self.data['Open'][i] and self.data['Close'][i] < self.data['Close'][i-1] and self.data['Close'][i-1] < self.data['Close'][i-2] and self.data['Close'][i-2] < self.data['Close'][i-3]:
                self.data.loc[i,'Bearish_Three_Line_Strike'] = 1
            elif self.data['Close'][i] > self.data['Open'][i] and self.data['Open'][i] < self.data['Close'][i-1] and self.data['Close'][i-2] > self.data['Open'][i-2] and self.data['Open'][i-2] > self.data['Close'][i-1] and self.data['Close'][i] > self.data['Close'][i-2]:
                self.data.loc[i,'Bullish_Abandoned_Baby'] = 1
            elif self.data['Close'][i] < self.data['Open'][i] and self.data['Open'][i] > self.data['Close'][i-1] and self.data['Close'][i-2] < self.data['Open'][i-2] and self.data['Open'][i-2] < self.data['Close'][i-1] and self.data['Close'][i] < self.data['Close'][i-2]:
                self.data.loc[i,'Bearish_Abandoned_Baby'] = 1
            elif self.data['Open'][i] < self.data['Close'][i-1] and self.data['Close'][i] > self.data['Open'][i] and self.data['Open'][i] < self.data['Close'][i-2] and self.data['Close'][i-2] < self.data['Open'][i-1]:
                self.data.loc[i,'Bullish_Kicker'] = 1
            elif self.data['Open'][i] > self.data['Close'][i-1] and self.data['Close'][i] < self.data['Open'][i] and self.data['Open'][i] > self.data['Close'][i-2] and self.data['Close'][i-2] > self.data['Open'][i-1]:
                self.data.loc[i,'Bearish_Kicker'] = 1
            elif self.data['Open'][i] < self.data['Close'][i-1] and self.data['Close'][i] > self.data['Open'][i] and self.data['Open'][i] > self.data['Close'][i-2] and self.data['Close'][i-2] > self.data['Open'][i-1]:
                self.data.loc[i,'Bullish_Kick_Backs'] = 1
            elif self.data['Open'][i] > self.data['Close'][i-1] and self.data['Close'][i] < self.data['Open'][i] and self.data['Open'][i] < self.data['Close'][i-2] and self.data['Close'][i-2] < self.data['Open'][i-1]:
                self.data.loc[i,'Bearish_Kick_Backs'] = 1
            elif self.data['Open'][i] > self.data['Close'][i-1] and self.data['Close'][i] > self.data['Open'][i] and self.data['Open'][i] < self.data['Close'][i-2] and self.data['Close'][i-2] < self.data['Open'][i-1]:
                self.data.loc[i,'Bullish_Harami_Cross'] = 1
            elif self.data['Open'][i] < self.data['Close'][i-1] and self.data['Close'][i] < self.data['Open'][i] and self.data['Open'][i] > self.data['Close'][i-2] and self.data['Close'][i-2] > self.data['Open'][i-1]:
                self.data.loc[i,'Bearish_Harami_Cross'] = 1
            elif self.data['Open'][i] < self.data['Close'][i-1] and self.data['Close'][i] > self.data['Close'][i-1]:
                self.data.loc[i,'Bullish_Engulfing_Pattern'] = 1
            elif self.data['Open'][i] > self.data['Close'][i-1] and self.data['Close'][i] < self.data['Close'][i-1]:
                self.data.loc[i,'Bearish_Engulfing_Pattern'] = 1

        return self.data

    
    def add_ADX(self):
        """
        Adds the Average Directional Movement Index to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the ADX column
        """
        #Calculate the +DI and -DI
        self.data['PDM'] = self.data['High'].diff(1)
        self.data['MDM'] = self.data['Low'].diff(1)
        self.data['TR'] = self.data[['High', 'Low', 'Close']].max(axis=1).diff(1)
        self.data['PDI'] = (self.data['PDM']/self.data['TR']) * 100
        self.data['MDI'] = (self.data['MDM']/self.data['TR']) * 100
 
        #Calculate the ADX
        self.data['ADX'] = 0.0
        for i in range(2, len(self.data)):
            self.data.at[i, 'ADX'] = self.data.at[i-1, 'ADX'] * (14 - 1)
            self.data.at[i, 'ADX'] += abs(self.data.at[i, 'PDI'] - self.data.at[i, 'MDI'])
            self.data.at[i, 'ADX'] /= 14
            self.data.at[i, 'ADX'] += abs(self.data.at[i, 'PDI'] - self.data.at[i, 'MDI']) / 2
 
        return self.data

# -Add On-Balance Volume

    def add_OBV(self):
        """
        Adds the On-Balance Volume column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the OBV column
        """
        self.data['OBV'] = 0.0
        for i in range(1, len(self.data)):
            if self.data.at[i, 'Close'] > self.data.at[i-1, 'Close']:
                self.data.at[i, 'OBV'] = self.data.at[i-1, 'OBV'] + self.data.at[i, 'Volume']
            elif self.data.at[i, 'Close'] < self.data.at[i-1, 'Close']:
                self.data.at[i, 'OBV'] = self.data.at[i-1, 'OBV'] - self.data.at[i, 'Volume']
            else:
                self.data.at[i, 'OBV'] = self.data.at[i-1, 'OBV']
 
        return self.data

    def add_momentum_indicators(self):
        """
        Adds the Momentum Indicator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Momentum Indicator columns
        """
        self.data['Momentum_12'] = self.data['Close'].diff(12)
        self.data['Momentum_26'] = self.data['Close'].diff(26)

        return self.data

    def add_rsi_divergence(self):
        """
        Adds the Relative Strength Index Divergence columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the RSI Divergence columns
        """
        self.data['RSI_Divergence_12'] = self.data['Momentum_12'].rolling(12).mean()
        self.data['RSI_Divergence_26'] = self.data['Momentum_26'].rolling(26).mean()

        return self.data

    def add_parabolic_sar(self):
        """
        Adds the Parabolic SAR columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Parabolic SAR columns
        """
        self.data['Parabolic_SAR_12'] = self.data['Close'].shift(12)
        self.data['Parabolic_SAR_26'] = self.data['Close'].shift(26)

        return self.data

    def add_bollinger_band_width(self):
        """
        Adds the Bollinger Band Width columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Bollinger Band Width columns
        """
        self.data['Bollinger_Band_Width_12'] = self.data['Upper_Bollinger_Band'].shift(12) - self.data['Lower_Bollinger_Band'].shift(12)
        self.data['Bollinger_Band_Width_26'] = self.data['Upper_Bollinger_Band'].shift(26) - self.data['Lower_Bollinger_Band'].shift(26)

        return self.data


    def add_ichimoku_kinko_hyo(self):
        """
        Adds the Ichimoku Kinko Hyo columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Ichimoku Kinko Hyo columns
        """
        self.data['Tenkan_Sen'] = self.data['High'].rolling(9).mean()
        self.data['Kijun_Sen'] = self.data['Low'].rolling(26).mean()
        self.data['Senkou_Span_A'] = (self.data['Tenkan_Sen'] + self.data['Kijun_Sen']) / 2
        self.data['Senkou_Span_B'] = self.data['Low'].rolling(52).mean()
        self.data['Chikou_Span'] = self.data['Close'].shift(26)

        return self.data

    def add_elliott_wave_analysis(self):
        """
        Adds the Elliott Wave Analysis columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Elliott Wave Analysis columns
        """
        self.data['Impulse_Waves'] = self.data['Close'].rolling(5).mean()
        self.data['Corrective_Waves'] = self.data['Close'].rolling(3).mean()
        self.data['Extension_Waves'] = self.data['Close'].rolling(7).mean()
        self.data['Corrective_Extension_Waves'] = self.data['Close'].rolling(9).mean()

        return self.data

    def add_moving_average_convergence_divergence(self):
        """
        Adds the Moving Average Convergence Divergence columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Moving Average Convergence Divergence columns
        """
        self.data['MACD_12_26'] = self.data['Close'].ewm(span=12).mean() - self.data['Close'].ewm(span=26).mean()
        self.data['MACD_Signal_9'] = self.data['MACD_12_26'].ewm(span=9).mean()
        self.data['MACD_Histogram'] = self.data['MACD_12_26'] - self.data['MACD_Signal_9']

        return self.data

    def add_on_balance_volume(self):
        """
        Adds the On-Balance Volume columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the On-Balance Volume columns
        """
        self.data['OBV_Change'] = np.where(self.data['Close'] > self.data['Close'].shift(1), self.data['Volume'],
                                        -self.data['Volume'])
        self.data['On-Balance_Volume'] = self.data['OBV_Change'].cumsum()

        return self.data

    def add_stochRSI(self):
        """
        Adds the StochRSI columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the StochRSI columns
        """
        self.data['StochRSI_K'] = 100 * (self.data['Close'] - self.data['Close'].rolling(14).min()) / (
                self.data['Close'].rolling(14).max() - self.data['Close'].rolling(14).min())
        self.data['StochRSI_D'] = self.data['StochRSI_K'].rolling(3).mean()

        return self.data

    def add_keltner_channels(self):
        """
        Adds the Keltner Channels columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Keltner Channels columns
        """
        self.data['Keltner_Channel_Mid'] = self.data['Close'].rolling(10).mean()
        self.data['Keltner_Channel_Upper'] = self.data['Keltner_Channel_Mid'] + 2 * self.data['Close'].rolling(10).std()
        self.data['Keltner_Channel_Lower'] = self.data['Keltner_Channel_Mid'] - 2 * self.data['Close'].rolling(10).std()

        return self.data

    def add_gann_fan(self):
        """
        Adds the Gann Fan columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Gann Fan columns
        """
        self.data['Gann_Fan_1'] = self.data['Close'] + (self.data['Close'].shift(1) - self.data['Close']) * 0.5
        self.data['Gann_Fan_2'] = self.data['Close'] + (self.data['Close'].shift(1) - self.data['Close']) * 0.382
        self.data['Gann_Fan_3'] = self.data['Close'] + (self.data['Close'].shift(1) - self.data['Close']) * 0.236
        self.data['Gann_Fan_4'] = self.data['Close'] + (self.data['Close'].shift(1) - self.data['Close']) * 0.618
        self.data['Gann_Fan_5'] = self.data['Close'] + (self.data['Close'].shift(1) - self.data['Close']) * 1
        self.data['Gann_Fan_6'] = self.data['Close'] + (self.data['Close'].shift(1) - self.data['Close']) * 1.382
        self.data['Gann_Fan_7'] = self.data['Close'] + (self.data['Close'].shift(1) - self.data['Close']) * 1.618

        return self.data


    def add_triple_exponential_average(self):
        """
        Adds the Triple Exponential Average columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Triple Exponential Average columns
        """
        self.data['Triple_EMA_5'] = self.data['Close'].ewm(span=5, adjust=False).mean()
        self.data['Triple_EMA_10'] = self.data['Close'].ewm(span=10, adjust=False).mean()
        self.data['Triple_EMA_20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
        self.data['Triple_EMA_50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        self.data['Triple_EMA_100'] = self.data['Close'].ewm(span=100, adjust=False).mean()
        self.data['Triple_EMA_200'] = self.data['Close'].ewm(span=200, adjust=False).mean()
        
        return self.data

    def add_rate_of_change(self):
        """
        Adds the Rate of Change columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Rate of Change columns
        """
        self.data['Rate_of_Change_5'] = self.data['Close'].pct_change(5).fillna(0)
        self.data['Rate_of_Change_10'] = self.data['Close'].pct_change(10).fillna(0)
        self.data['Rate_of_Change_20'] = self.data['Close'].pct_change(20).fillna(0)
        self.data['Rate_of_Change_50'] = self.data['Close'].pct_change(50).fillna(0)
        self.data['Rate_of_Change_100'] = self.data['Close'].pct_change(100).fillna(0)
        self.data['Rate_of_Change_200'] = self.data['Close'].pct_change(200).fillna(0)

        return self.data

    def add_detrended_price_oscillator(self):
        """
        Adds the Detrended Price Oscillator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Detrended Price Oscillator columns
        """
        self.data['DPO_5'] = (self.data['Close'] - self.data['Close'].shift(int(5 / 2 + 1))) / self.data['Close'].rolling(5).mean()
        self.data['DPO_10'] = (self.data['Close'] - self.data['Close'].shift(int(10 / 2 + 1))) / self.data['Close'].rolling(10).mean()
        self.data['DPO_20'] = (self.data['Close'] - self.data['Close'].shift(int(20 / 2 + 1))) / self.data['Close'].rolling(20).mean()
        self.data['DPO_50'] = (self.data['Close'] - self.data['Close'].shift(int(50 / 2 + 1))) / self.data['Close'].rolling(50).mean()
        self.data['DPO_100'] = (self.data['Close'] - self.data['Close'].shift(int(100 / 2 + 1))) / self.data['Close'].rolling(100).mean()
        self.data['DPO_200'] = (self.data['Close'] - self.data['Close'].shift(int(200 / 2 + 1))) / self.data['Close'].rolling(200).mean()
        
        return self.data

    def add_mass_index(self):
        """
        Adds the Mass Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Mass Index columns
        """
        self.data['Mass_Index_9'] = 0
        self.data['Mass_Index_25'] = 0

        for i in range(9, len(self.data)):
            self.data.loc[i, 'Mass_Index_9'] = (self.data.loc[i-9:i, 'High'].max() - self.data.loc[i-9:i, 'Low'].min()) / (self.data.loc[i-9:i, 'High'].max() - self.data.loc[i-9:i, 'Low'].min())

        for i in range(25, len(self.data)):
            self.data.loc[i, 'Mass_Index_25'] = (self.data.loc[i-25:i, 'High'].max() - self.data.loc[i-25:i, 'Low'].min()) / (self.data.loc[i-25:i, 'High'].max() - self.data.loc[i-25:i, 'Low'].min())
            
        return self.data


    def add_vortex_indicator(self):
        """
        Adds the Vortex Indicator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Vortex Indicator columns
        """
        self.data['VI_Positive'] = self.data['High'].diff().fillna(0.0000) - self.data['Low'].diff().fillna(0.0000)
        self.data['VI_Negative'] = abs(self.data['High'].diff().fillna(0.0000)) - abs(self.data['Low'].diff().fillna(0.0000))
        self.data['VI_Positive'] = self.data['VI_Positive'].rolling(14).sum()
        self.data['VI_Negative'] = self.data['VI_Negative'].rolling(14).sum()
        self.data['Vortex_Indicator'] = self.data['VI_Positive'] / self.data['VI_Negative']

        return self.data

    def add_kst_oscillator(self):
        """
        Adds the KST Oscillator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the KST Oscillator columns
        """
        self.data['KST_10'] = self.data['Close'].rolling(10).mean().diff(10)
        self.data['KST_15'] = self.data['Close'].rolling(15).mean().diff(15)
        self.data['KST_20'] = self.data['Close'].rolling(20).mean().diff(20)
        self.data['KST_30'] = self.data['Close'].rolling(30).mean().diff(30)
        self.data['KST_50'] = self.data['Close'].rolling(50).mean().diff(50)
        self.data['KST_Oscillator'] = self.data['KST_10'] + 2 * self.data['KST_15'] + 3 * self.data['KST_20'] + 4 * self.data['KST_30'] + 5 * self.data['KST_50']

        return self.data

    def add_chaikin_money_flow(self):
        """
        Adds the Chaikin Money Flow columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Chaikin Money Flow columns
        """
        self.data['Money_Flow_Multiplier'] = ((self.data['Close'] - self.data['Low']) - (self.data['High'] - self.data['Close'])) / (self.data['High'] - self.data['Low'])
        self.data['Money_Flow_Volume'] = self.data['Money_Flow_Multiplier'] * self.data['Volume']
        self.data['Chaikin_Money_Flow'] = self.data['Money_Flow_Volume'].rolling(20).sum() / self.data['Volume'].rolling(20).sum()

        return self.data

    def add_average_directional_index(self):
        """
        Adds the Average Directional Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Average Directional Index columns
        """
        self.data['High_Minus_Low'] = self.data['High'] - self.data['Low']
        self.data['High_Minus_Close_Previous'] = self.data['High'] - self.data['Close'].shift(1)
        self.data['Close_Previous_Minus_Low'] = self.data['Close'].shift(1) - self.data['Low']
        self.data['True_Range'] = self.data[['High_Minus_Low', 'High_Minus_Close_Previous', 'Close_Previous_Minus_Low']].max(axis=1)
        self.data['DM_Plus'] = 0
        self.data['DM_Minus'] = 0
        self.data.loc[(self.data['High_Minus_Low'] > 0) & (self.data['High_Minus_Close_Previous'] > 0) & (self.data['High_Minus_Low'] > self.data['High_Minus_Close_Previous']), 'DM_Plus'] = self.data['High_Minus_Low']
        self.data.loc[(self.data['High_Minus_Low'] > 0) & (self.data['Close_Previous_Minus_Low'] > 0) & (self.data['High_Minus_Low'] > self.data['Close_Previous_Minus_Low']), 'DM_Minus'] = self.data['High_Minus_Low']
        self.data['DI_Plus'] = self.data['DM_Plus'].rolling(14).sum() / self.data['True_Range'].rolling(14).sum()
        self.data['DI_Minus'] = self.data['DM_Minus'].rolling(14).sum() / self.data['True_Range'].rolling(14).sum()
        self.data['Average_Directional_Index'] = 100 * (abs(self.data['DI_Plus'] - self.data['DI_Minus']) / (self.data['DI_Plus'] + self.data['DI_Minus']))

        return self.data

    def add_ultimate_oscillator(self):
        """
        Adds the Ultimate Oscillator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Ultimate Oscillator columns
        """
        self.data['Total_TR'] = 0
        self.data['BP'] = 0
        self.data.loc[self.data['Close'] > self.data['Close'].shift(1), 'BP'] = self.data['Close'] - self.data['Close'].shift(1)
        self.data.loc[self.data['Close'] < self.data['Close'].shift(1), 'BP'] = 0
        self.data.loc[self.data['High'] > self.data['High'].shift(1), 'Total_TR'] = self.data['High'] - self.data['Low']
        self.data.loc[self.data['High'] < self.data['High'].shift(1), 'Total_TR'] = self.data['High'] - self.data['Close'].shift(1)
        self.data.loc[self.data['Low'] < self.data['Low'].shift(1), 'Total_TR'] = self.data['Low'] - self.data['Close'].shift(1)
        self.data['Average_7_TR'] = self.data['Total_TR'].rolling(7).sum() / 7
        self.data['Average_14_TR'] = self.data['Total_TR'].rolling(14).sum() / 14
        self.data['Average_28_TR'] = self.data['Total_TR'].rolling(28).sum() / 28
        self.data['Raw_UO'] = 100 * ((4 * self.data['Average_7_TR']) + (2 * self.data['Average_14_TR']) + self.data['Average_28_TR']) / (4 * self.data['Total_TR'])
        self.data['Average_7_BP'] = self.data['BP'].rolling(7).sum() / 7
        self.data['Average_14_BP'] = self.data['BP'].rolling(14).sum() / 14
        self.data['Average_28_BP'] = self.data['BP'].rolling(28).sum() / 28
        self.data['Ultimate_Oscillator'] = 100 * ((4 * self.data['Average_7_BP']) + (2 * self.data['Average_14_BP']) + self.data['Average_28_BP']) / (4 * self.data['Total_TR'])

        return self.data

    def add_linear_regression_indicator(self):
        """
        Adds the Linear Regression Indicator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Linear Regression Indicator columns
        """
        self.data['Linear_Regression_Slope'] = self.data['Close'].rolling(14).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
        self.data['Linear_Regression_Intercept'] = self.data['Close'].rolling(14).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[1], raw=True)
        self.data['Linear_Regression_Indicator'] = self.data['Close'] - (self.data['Linear_Regression_Slope'] * 14 + self.data['Linear_Regression_Intercept'])

        return self.data

    def add_negative_volume_index(self):
        """
        Adds the Negative Volume Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Negative Volume Index columns
        """
        self.data['Negative_Volume_Index'] = self.data['Volume'].diff().apply(lambda x: 1 if x < 0 else 0).cumsum()

        return self.data

    def add_mcclellan_oscillator(self):
        """
        Adds the McClellan Oscillator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the McClellan Oscillator columns
        """
        self.data['McClellan_Oscillator'] = self.data['Close'].diff(periods=19).rolling(19).mean() - self.data['Close'].diff(periods=39).rolling(39).mean()

        return self.data

    def add_kaufman_adaptive_moving_average(self):
        """
        Adds the Kaufman Adaptive Moving Average columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Kaufman Adaptive Moving Average columns
        """
        self.data['KAMA'] = self.data['Close'].ewm(com=2, min_periods=30, ignore_na=False, adjust=True).mean()

        return self.data

    def add_moving_average_of_oscillator(self):
        """
        Adds the Moving Average of Oscillator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Moving Average of Oscillator columns
        """
        self.data['MACD'] = self.data['Close'].ewm(span=12, adjust=False).mean() - self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD_Signal'] = self.data['MACD'].rolling(9).mean()

        return self.data

    def add_klinger_volume_oscillator(self):
        """
        Adds the Klinger Volume Oscillator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Klinger Volume Oscillator columns
        """
        self.data['KVO_High'] = self.data['Volume'].rolling(14).mean()
        self.data['KVO_Low'] = self.data['Volume'].shift(1).rolling(14).mean()
        self.data['Klinger_Volume_Oscillator'] = 100 * ((self.data['KVO_High'] - self.data['KVO_Low']) / (self.data['KVO_High'] + self.data['KVO_Low']))

        return self.data

    def add_coppock_curve(self):
        """
        Calculates and adds the Coppock Curve columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Coppock Curve columns
        """
        coppock_curve = []

        for i in range(1,len(self.data['Close'])):
            diff = abs(self.data['Close'][i] - self.data['Close'][i-1])
            coppock_curve.append(diff)
        
        self.data['Coppock_Curve'] = pd.Series(coppock_curve).rolling(11).mean()

        return self.data

    def add_elder_ray_index(self):
        """
        Adds the Elder Ray Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Elder Ray Index columns
        """
        self.data['Bull_Power'] = self.data['High'] - self.data['Close'].shift(1)
        self.data['Bear_Power'] = self.data['Low'] - self.data['Close'].shift(1)

        return self.data


    def add_force_index(self):
        """
        Adds the Force Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Force Index columns
        """
        self.data['Force_Index'] = (self.data['Close'] - self.data['Close'].shift(1)) * self.data['Volume']
        self.data['Force_Index_EMA'] = self.data['Force_Index'].ewm(span=13).mean()

        return self.data


    def add_elder_impulse_system(self):
        """
        Adds the Elder Impulse System columns to the DataFrame

        Parameters
        ----------
        data : DataFrame
            A pandas DataFrame containing the stock data

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Elder Impulse System columns
        """
        self.data['12-period EMA'] = self.data['Close'].ewm(span=12).mean()
        self.data['26-period EMA'] = self.data['Close'].ewm(span=26).mean()
        self.data['Signal'] = self.data['12-period EMA'] - self.data['26-period EMA']
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal']
        self.data['MACD_Histogram_EMA'] = self.data['MACD_Histogram'].ewm(span=13).mean()
        self.data['MACD_Histogram_EMA_2'] = self.data['MACD_Histogram_EMA'].ewm(span=8).mean()

        return self.data

    def add_chandelier_exit(self):
        """
        Adds the Chandelier Exit columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Chandelier Exit columns
        """
        self.data['ATR'] = self.data['High'] - self.data['Low']
        self.data['ATR_EMA'] = self.data['ATR'].ewm(span=13).mean()
        self.data['Chandelier_Exit'] = self.data['Close'].shift(1) - (self.data['ATR_EMA'] * 3)

        return self.data

    def add_keltner_channel_breakout_system(self):
        """
        Adds the Keltner Channel Breakout System columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Keltner Channel Breakout System columns
        """
        self.data['Upper_Keltner_Channel'] = self.data['Close'].rolling(20).mean() + (self.data['Close'].rolling(20).std() * 2)
        self.data['Lower_Keltner_Channel'] = self.data['Close'].rolling(20).mean() - (self.data['Close'].rolling(20).std() * 2)

        return self.data
    
    def add_parabolic_sar_adx_combo(self):
        """
        Adds the Parabolic SAR and ADX Combo columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Parabolic SAR and ADX Combo columns
        """
        self.data['Parabolic_SAR'] = self.data['Close'].shift(1).rolling(2).mean()
        self.data['ADX'] = self.data['Close'].rolling(3).mean()

        return self.data
    
    def add_dmi_stochastic_system(self):
        """
        Adds the DMI Stochastic System columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the DMI Stochastic System columns
        """
        self.data['High_Minus_Low'] = self.data['High'] - self.data['Low']
        self.data['High_Minus_Previous_Close'] = self.data['High'] - self.data['Close'].shift(1)
        self.data['Low_Minus_Previous_Close'] = self.data['Low'] - self.data['Close'].shift(1)
        self.data['DMI_Plus'] = self.data[['High_Minus_Low', 'High_Minus_Previous_Close']].max(axis=1)
        self.data['DMI_Minus'] = self.data[['High_Minus_Low', 'Low_Minus_Previous_Close']].min(axis=1)
        self.data['DMI_Plus_MA'] = self.data['DMI_Plus'].rolling(14).mean()
        self.data['DMI_Minus_MA'] = self.data['DMI_Minus'].rolling(14).mean()
        self.data['DMI_Diff'] = self.data['DMI_Plus_MA'] - self.data['DMI_Minus_MA']
        self.data['DMI_Diff_MA'] = self.data['DMI_Diff'].rolling(14).mean()

        return self.data
    
    def add_swing_index(self):
        """
        Adds the Swing Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Swing Index columns
        """
        self.data['Swing_Index'] = (self.data['High'] + self.data['Low'] + 2 * self.data['Close']) / 4
        self.data['Swing_Index_MA'] = self.data['Swing_Index'].rolling(3).mean()

        return self.data
    
    def add_andrews_pitchfork(self):
        """
        Adds the Andrews Pitchfork columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Andrews Pitchfork columns
        """
        self.data['Andrews_Pitchfork_Line_1'] = self.data['Close'].shift(2) + (2 * (self.data['High'] - self.data['Low']))
        self.data['Andrews_Pitchfork_Line_2'] = self.data['Close'].shift(2) - (2 * (self.data['High'] - self.data['Low']))
        self.data['Andrews_Pitchfork_Line_3'] = self.data['Close'].shift(2)
        self.data['Andrews_Pitchfork_Line_1_MA'] = self.data['Andrews_Pitchfork_Line_1'].rolling(3).mean()
        self.data['Andrews_Pitchfork_Line_2_MA'] = self.data['Andrews_Pitchfork_Line_2'].rolling(3).mean()
        self.data['Andrews_Pitchfork_Line_3_MA'] = self.data['Andrews_Pitchfork_Line_3'].rolling(3).mean()

        return self.data

    def add_ichimoku_cloud(self):
        """
        Adds the Ichimoku Cloud columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Ichimoku Cloud columns
        """
        self.data['Tenkan_Sen'] = self.data['High'].rolling(9).mean()
        self.data['Kijun_Sen'] = self.data['Low'].rolling(26).mean()
        self.data['Senkou_Span_A'] = ((self.data['Tenkan_Sen'] + self.data['Kijun_Sen'])/2).shift(26)
        self.data['Senkou_Span_B'] = self.data['Low'].rolling(52).mean().shift(26)
        self.data['Chikou_Span'] = self.data['Close'].shift(26)

        return self.data

    def add_relative_strength_index_wilder_smoothing(self):
        """
        Adds the Relative Strength Index Wilder Smoothing columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the RSI Wilder Smoothing columns
        """
        self.data['Change'] = self.data['Close'].diff(1)
        self.data['Gain'] = self.data['Change'].mask(self.data['Change'] < 0, 0)
        self.data['Loss'] = self.data['Change'].mask(self.data['Change'] > 0, 0)
        self.data['Avg_Gain'] = self.data['Gain'].ewm(com=14).mean()
        self.data['Avg_Loss'] = self.data['Loss'].ewm(com=14).mean()
        self.data['RS'] = self.data['Avg_Gain'] / self.data['Avg_Loss']
        self.data['RSI'] = 100 - (100 / (1 + self.data['RS']))

        return self.data

    def add_rainbow_charts(self):
        """
        Adds the Rainbow Chart columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Rainbow Chart columns
        """
        self.data['RSI_5'] = self.data['Close'].rolling(5).mean()
        self.data['RSI_14'] = self.data['Close'].rolling(14).mean()
        self.data['RSI_21'] = self.data['Close'].rolling(21).mean()
        self.data['RSI_28'] = self.data['Close'].rolling(28).mean()
        self.data['RSI_35'] = self.data['Close'].rolling(35).mean()

        return self.data
    
    def add_market_profile(self):
        """
        Adds the Market Profile columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Market Profile columns
        """
        self.data['High_Volume'] = self.data['High'].rolling(20).apply(lambda x: np.percentile(x, 95), raw=True)
        self.data['Low_Volume'] = self.data['Low'].rolling(20).apply(lambda x: np.percentile(x, 5), raw=True)
        self.data['High_Volume_Average'] = self.data['High_Volume'].rolling(5).mean()
        self.data['Low_Volume_Average'] = self.data['Low_Volume'].rolling(5).mean()

        return self.data

    def add_elder_safe_zone_strategy(self):
        """
        Adds the Elder Safe Zone Strategy columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Elder Safe Zone Strategy columns
        """
        self.data['EMA_13'] = self.data['Close'].ewm(span=13).mean()
        self.data['EMA_34'] = self.data['Close'].ewm(span=34).mean()
        self.data['MACD'] = self.data['EMA_13'] - self.data['EMA_34']
        self.data['Signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['Histogram'] = self.data['MACD'] - self.data['Signal']
        self.data['Safe_Zone'] = self.data['Histogram'].rolling(3).apply(lambda x: np.percentile(x, 0.5), raw=True)

        return self.data

    def add_moving_average_envelope(self):
        """
        Adds the Moving Average Envelope columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Moving Average Envelope columns
        """
        self.data['MA_20_Upper'] = self.data['Close'].rolling(20).mean() + 0.03 * self.data['Close'].rolling(20).std()
        self.data['MA_20_Lower'] = self.data['Close'].rolling(20).mean() - 0.03 * self.data['Close'].rolling(20).std()
        self.data['MA_50_Upper'] = self.data['Close'].rolling(50).mean() + 0.03 * self.data['Close'].rolling(50).std()
        self.data['MA_50_Lower'] = self.data['Close'].rolling(50).mean() - 0.03 * self.data['Close'].rolling(50).std()
        self.data['MA_200_Upper'] = self.data['Close'].rolling(200).mean() + 0.03 * self.data['Close'].rolling(200).std()
        self.data['MA_200_Lower'] = self.data['Close'].rolling(200).mean() - 0.03 * self.data['Close'].rolling(200).std()

        return self.data

    def add_bollinger_band_squeeze(self):
        """
        Adds the Bollinger Band Squeeze columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Bollinger Band Squeeze columns
        """
        self.data['BB_Upper'] = self.data['Close'].rolling(20).mean() + 2 * self.data['Close'].rolling(20).std()
        self.data['BB_Lower'] = self.data['Close'].rolling(20).mean() - 2 * self.data['Close'].rolling(20).std()
        self.data['BB_Squeeze'] = self.data['BB_Upper'] - self.data['BB_Lower']

        return self.data

    def add_obvios_momentum_indicator(self):
        """
        Adds the OBVios Momentum Indicator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the OBVios Momentum Indicator columns
        """
        self.data['OBVios_Momentum'] = (self.data['Close'] - self.data['Close'].shift(1)) * self.data['Volume']
        return self.data

    def add_keltner_channel_breakout_system(self):
        """
        Adds the Keltner Channel Breakout System columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Keltner Channel Breakout System columns
        """
        self.data['Keltner_Channel_Upper_Band'] = self.data['Close'].rolling(20).mean() + 2 * self.data['Close'].rolling(20).std() 
        self.data['Keltner_Channel_Lower_Band'] = self.data['Close'].rolling(20).mean() - 2 * self.data['Close'].rolling(20).std()
        return self.data

    def add_money_flow_index(self):
        """
        Adds the Money Flow Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Money Flow Index columns
        """
        self.data['Money_Flow_Index'] = ((self.data['High'] - self.data['Low']) - (self.data['High'].shift(1) - self.data['Low'].shift(1))) / (self.data['High'] - self.data['Low'])
        return self.data

    def add_donchian_channel_breakout_system(self):
        """
        Adds the Donchian Channel Breakout System columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Donchian Channel Breakout System columns
        """
        self.data['Donchian_Channel_Upper_Band'] = self.data['High'].rolling(20).max()
        self.data['Donchian_Channel_Lower_Band'] = self.data['Low'].rolling(20).min()
        return self.data

    def add_average_directional_movement_index(self):
        """
        Adds the Average Directional Movement Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Average Directional Movement Index columns
        """
        self.data['Average_Directional_Movement_Index'] = (self.data['High'] - self.data['High'].shift(1)) / (self.data['High'] - self.data['Low'])
        return self.data

    def add_commodity_channel_index(self):
        """
        Adds the Commodity Channel Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Commodity Channel Index columns
        """
        self.data['Commodity_Channel_Index'] = (self.data['Close'] - self.data['Close'].rolling(20).mean()) / (0.015 * self.data['Close'].rolling(20).std())
        return self.data

    def add_stochastic_rsi(self):
        """
        Adds the Stochastic RSI columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Stochastic RSI columns
        """
        self.data['Stochastic_RSI'] = (self.data['Close'] - self.data['Low'].rolling(14).min()) / (self.data['High'].rolling(14).max() - self.data['Low'].rolling(14).min())
        return self.data

    def add_volume_weighted_average_price(self):
        """
        Adds the Volume Weighted Average Price columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Volume Weighted Average Price columns
        """
        self.data['Volume_Weighted_Average_Price'] = (self.data['Close'] * self.data['Volume']) / (self.data['Volume'].rolling(20).sum())
        return self.data
    

    def add_connors_rsi(self):
        """
        Adds the Connors RSI indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Connors RSI column
        """
        self.data['Connors_RSI'] = self.data['Close'].rolling(3).mean() + (2 * self.data['Close'].diff(1).abs().rolling(3).mean()) + (3 * self.data['Close'].diff(2).abs().rolling(3).mean())

        return self.data

    def add_super_trend_indicator(self):
        """
        Adds the SuperTrend indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the SuperTrend column
        """
        self.data['Super_Trend'] = (self.data['High'] + self.data['Low']) / 2 + (self.data['High'] - self.data['Low']) * 0.5
        
        return self.data


    def add_average_true_range_stop(self):
        """
        Adds the Average True Range Stop indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Average True Range Stop column
        """
        self.data['ATR_Stop'] = self.data['Close'].rolling(14).mean() + (2 * self.data['Close'].diff(1).abs().rolling(14).mean())
        
        return self.data

    def add_swing_index(self):
        """
        Adds the Swing Index indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Swing Index column
        """
        self.data['Swing_Index'] = (self.data['High'] - self.data['Low'].shift(1)) / (self.data['High'] - self.data['Low'])

        return self.data

    def add_ichimoku_kinko_hyo(self):
        """
        Adds the Ichimoku Kinko Hyo indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Ichimoku Kinko Hyo columns
        """
        self.data['Ichimoku_Kinko_Hyo_Tenkan_Sen'] = self.data['High'].rolling(9).mean()
        self.data['Ichimoku_Kinko_Hyo_Kijun_Sen'] = self.data['Low'].rolling(26).mean()
        self.data['Ichimoku_Kinko_Hyo_Senkou_Span_A'] = ((self.data['High'].rolling(9).mean() + self.data['Low'].rolling(26).mean()) / 2).shift(26)
        self.data['Ichimoku_Kinko_Hyo_Senkou_Span_B'] = ((self.data['High'].rolling(26).max() + self.data['Low'].rolling(26).min()) / 2).shift(26)
        self.data['Ichimoku_Kinko_Hyo_Chikou_Span'] = self.data['Close'].shift(26)

        return self.data

    def add_keltner_channel_volatility_breakout_system(self):
        """
        Adds the Keltner Channel Volatility Breakout System indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Keltner Channel Volatility Breakout System columns
        """
        self.data['KC_Upper_Band'] = self.data['Close'].rolling(20).mean() + (2 * self.data['Close'].rolling(20).std())
        self.data['KC_Lower_Band'] = self.data['Close'].rolling(20).mean() - (2 * self.data['Close'].rolling(20).std())
        self.data['KC_Mid_Band'] = self.data['Close'].rolling(20).mean()

        return self.data

    def add_triple_screen_trading_system(self):
        """
        Adds the Triple Screen Trading System indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Triple Screen Trading System columns
        """
        self.data['TS_First_Screen'] = self.data['Close'].rolling(20).mean()
        self.data['TS_Second_Screen'] = self.data['Close'].rolling(50).mean()
        self.data['TS_Third_Screen'] = self.data['Close'].rolling(200).mean()

        return self.data

    def add_standard_deviation_channel(self):
        """
        Adds the Standard Deviation Channel indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Standard Deviation Channel columns
        """
        self.data['SDC_Upper_Band'] = self.data['Close'].rolling(20).mean() + (2 * self.data['Close'].rolling(20).std())
        self.data['SDC_Lower_Band'] = self.data['Close'].rolling(20).mean() - (2 * self.data['Close'].rolling(20).std())
        self.data['SDC_Mid_Band'] = self.data['Close'].rolling(20).mean()

        return self.data

    def add_adaptive_moving_average(self):
        """
        Adds the Adaptive Moving Average indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Adaptive Moving Average column
        """
        self.data['Adaptive_MA'] = self.data['Close'].ewm(com=2).mean()

        return self.data

    def add_chande_momentum_oscillator(self):
        """
        Adds the Chande Momentum Oscillator indicator to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Chande Momentum Oscillator column
        """
        self.data['CMO'] = (self.data['Close'].diff().shift(1) / (self.data['High'] - self.data['Low']).shift(1)).ewm(span=14).mean() * 100

        return self.data


    def add_volatility_stop(self):
        """
        Adds the Volatility Stop column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Volatility Stop column
        """
        self.data['Volatility_Stop'] = self.data['Close'].rolling(20).std() 

        return self.data

    def add_volume_zone_oscillator(self):
        """
        Adds the Volume Zone Oscillator column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Volume Zone Oscillator column
        """
        self.data['Volume_Zone_Oscillator'] = self.data['Volume'].rolling(20).mean() 

        return self.data

    def add_vama(self):
        """
        Adds the Volatility Adjusted Moving Average columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Volatility Adjusted Moving Average columns
        """
        self.data['VAMA_5'] = self.data['Close'].rolling(5).mean() * self.data['Close'].rolling(20).std() 
        self.data['VAMA_10'] = self.data['Close'].rolling(10).mean() * self.data['Close'].rolling(20).std() 
        self.data['VAMA_20'] = self.data['Close'].rolling(20).mean() * self.data['Close'].rolling(20).std() 
        self.data['VAMA_50'] = self.data['Close'].rolling(50).mean() * self.data['Close'].rolling(20).std() 
        self.data['VAMA_100'] = self.data['Close'].rolling(100).mean() * self.data['Close'].rolling(20).std() 
        self.data['VAMA_200'] = self.data['Close'].rolling(200).mean() * self.data['Close'].rolling(20).std() 

        return self.data

    def add_acceleration_deceleration_oscillator(self):
        """
        Adds the Acceleration/Deceleration Oscillator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Acceleration/Deceleration Oscillator columns
        """
        self.data['AccDecel_Oscillator_5'] = self.data['Close'].rolling(5).mean() - self.data['Close'].rolling(35).mean()
        self.data['AccDecel_Oscillator_10'] = self.data['Close'].rolling(10).mean() - self.data['Close'].rolling(35).mean()
        self.data['AccDecel_Oscillator_20'] = self.data['Close'].rolling(20).mean() - self.data['Close'].rolling(35).mean()
        self.data['AccDecel_Oscillator_50'] = self.data['Close'].rolling(50).mean() - self.data['Close'].rolling(35).mean()
        self.data['AccDecel_Oscillator_100'] = self.data['Close'].rolling(100).mean() - self.data['Close'].rolling(35).mean()
        self.data['AccDecel_Oscillator_200'] = self.data['Close'].rolling(200).mean() - self.data['Close'].rolling(35).mean()

        return self.data

    def add_directional_movement_index(self):
        """
        Adds the Directional Movement Index columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Directional Movement Index columns
        """
        self.data['DI_Plus'] = self.data['High'].diff()
        self.data['DI_Minus'] = -self.data['Low'].diff()
        self.data['DI_Plus'] = self.data['DI_Plus'].clip(lower=0)
        self.data['DI_Minus'] = self.data['DI_Minus'].clip(lower=0)
        self.data['DI_Plus_EMA'] = self.data['DI_Plus'].ewm(span=14, adjust=False).mean()
        self.data['DI_Minus_EMA'] = self.data['DI_Minus'].ewm(span=14, adjust=False).mean()
        self.data['DI_Diff'] = self.data['DI_Plus_EMA'] - self.data['DI_Minus_EMA']
        self.data['DI_Sum'] = self.data['DI_Plus_EMA'] + self.data['DI_Minus_EMA']
        self.data['DX'] = np.where(self.data['DI_Sum'] > 0, self.data['DI_Diff']/self.data['DI_Sum'], 0)
        self.data['ADX'] = self.data['DX'].ewm(span=14, adjust=False).mean()

        return self.data

    def add_elder_ray_bull_bear_power(self):
        """
        Adds the Elder-Ray Bull/Bear Power columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Elder-Ray Bull/Bear Power columns
        """
        self.data['Bull_Power'] = self.data['High'] - self.data['Close'].shift(1)
        self.data['Bear_Power'] = self.data['Low'] - self.data['Close'].shift(1)
        self.data['Bull_Power_EMA'] = self.data['Bull_Power'].ewm(span=13, adjust=False).mean()
        self.data['Bear_Power_EMA'] = self.data['Bear_Power'].ewm(span=13, adjust=False).mean()

        return self.data

    def add_detrended_price_oscillator(self):
        """
        Adds the Detrended Price Oscillator column to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the Detrended Price Oscillator column
        """
        self.data['Detrended_Price_Oscillator'] = self.data['Close'].shift(1) - self.data['Close'].rolling(20).mean()

        return self.data

    def add_trix_indicator(self):
        """
        Adds the TRIX Indicator columns to the DataFrame

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the TRIX Indicator columns
        """
        self.data['TRIX_5'] = self.data['Close'].ewm(span=5, adjust=False).mean().ewm(span=5, adjust=False).mean().ewm(span=5, adjust=False).mean()
        self.data['TRIX_10'] = self.data['Close'].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
        self.data['TRIX_20'] = self.data['Close'].ewm(span=20, adjust=False).mean().ewm(span=20, adjust=False).mean().ewm(span=20, adjust=False).mean()
        self.data['TRIX_50'] = self.data['Close'].ewm(span=50, adjust=False).mean().ewm(span=50, adjust=False).mean().ewm(span=50, adjust=False).mean()
        self.data['TRIX_100'] = self.data['Close'].ewm(span=100, adjust=False).mean().ewm(span=100, adjust=False).mean().ewm(span=100, adjust=False).mean()
        self.data['TRIX_200'] = self.data['Close'].ewm(span=200, adjust=False).mean().ewm(span=200, adjust=False).mean().ewm(span=200, adjust=False).mean()

        return self.data
