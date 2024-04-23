import os
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
from binance.client import Client
from textblob import TextBlob
import talib

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown

# Retrieve API keys from environment variables
os.environ['BINANCE_API_KEY'] = 'dukQ3wMRSsv5CN2PImY5LwWaeXl8heOJmCKuRGOQRYarCrbTJWAV4FFHmeToKG3z'
os.environ['BINANCE_API_SECRET'] = 'IbNHHtbz3FjzyMV6nLqUW4tNt3Eo9Znq7BLlD6Xmql4M1ekottKCAJwzhjJzs87F'
os.environ['NEWS_API_KEY'] = 'cbd55698a7a54bc39dd9379ca9faf95e'

binance_api_key = os.environ.get('BINANCE_API_KEY')
binance_api_secret = os.environ.get('BINANCE_API_SECRET')
news_api_key = os.environ.get('NEWS_API_KEY')

# Check if API keys are available
if not (binance_api_key and binance_api_secret and news_api_key):
    raise ValueError("Please set environment variables for Binance and NewsAPI keys.")

# Initialize Binance client
client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

class StockPricePredictor(App):
    def build(self):
        # Layout
        layout = BoxLayout(orientation='vertical', padding=10)

        # Symbol dropdown
        symbol_layout = BoxLayout(orientation='horizontal', spacing=10)
        symbol_label = Label(text="Symbol:")
        self.symbol_dropdown = DropDown()
        self.symbol_input = Button(text='Select Symbol', size_hint=(None, None), size=(150, 44))
        self.symbol_input.bind(on_release=self.symbol_dropdown.open)
        symbol_dropdown_options = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
        for option in symbol_dropdown_options:
            btn = Button(text=option, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: self.select_symbol(btn.text))
            self.symbol_dropdown.add_widget(btn)
        symbol_layout.add_widget(symbol_label)
        symbol_layout.add_widget(self.symbol_input)

        # Time frame dropdown
        time_frame_layout = BoxLayout(orientation='horizontal', spacing=10)
        time_frame_label = Label(text="Timeframe:")
        self.time_frame_dropdown = DropDown()
        self.time_frame_input = Button(text='Select Timeframe', size_hint=(None, None), size=(150, 44))
        self.time_frame_input.bind(on_release=self.time_frame_dropdown.open)
        time_frame_dropdown_options = ['1 Hour', '5 Hours', '1 Day', '2 Days']
        for option in time_frame_dropdown_options:
            btn = Button(text=option, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: self.select_time_frame(btn.text))
            self.time_frame_dropdown.add_widget(btn)
        time_frame_layout.add_widget(time_frame_label)
        time_frame_layout.add_widget(self.time_frame_input)

        # Predict button
        predict_button = Button(text="Predict Price", on_press=self.predict_price)

        # Feedback text
        self.feedback_text = Label(text="Prediction feedback will appear here.")

        # Add widgets to layout
        layout.add_widget(symbol_layout)
        layout.add_widget(time_frame_layout)
        layout.add_widget(predict_button)
        layout.add_widget(self.feedback_text)

        return layout

    def select_symbol(self, symbol):
        self.symbol_input.text = symbol

    def select_time_frame(self, time_frame):
        # Extract only the numerical part of the time frame
        time_frame_value = ''.join(filter(str.isdigit, time_frame))
        self.time_frame_input.text = time_frame
        # Store the selected time frame value as an attribute
        self.selected_time_frame = time_frame
        # Store the numerical part of the time frame as an attribute
        self.selected_time_frame_value = time_frame_value

    def fetch_news(self, stock_symbol):
        url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={news_api_key}'
        response = requests.get(url)
        news_data = response.json()
        articles = news_data.get('articles', [])
        return articles

    def analyze_sentiment(self, articles):
        sentiments = []
        for article in articles:
            text = article.get('title', '') + ' ' + article.get('description', '')
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        return average_sentiment

    def fetch_historical_data(self, stock_symbol, time_frame_type, time_frame_value):
        # Initialize start_date with a default value
        start_date = datetime.now()

        if time_frame_type == '1 Hour':
            start_date -= timedelta(days=int(time_frame_value) * 7)  # Fetch past 7 days
        elif time_frame_type == '5 Hours':
            start_date -= timedelta(days=int(time_frame_value) * 7)  # Fetch past 7 days
        elif time_frame_type == '1 Day':
            start_date -= timedelta(days=int(time_frame_value) * 7)  # Fetch past 7 days
        elif time_frame_type == '2 Days':
            start_date -= timedelta(days=int(time_frame_value) * 7)  # Fetch past 7 days

        kline_interval = Client.KLINE_INTERVAL_1HOUR if time_frame_type in ['1 Hour', '5 Hours'] else Client.KLINE_INTERVAL_1DAY

        start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        historical_data = client.get_historical_klines(stock_symbol, kline_interval, start_date_str, end_date_str)

        if not historical_data:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

    def calculate_technical_indicators(self, df):
        try:
            df['SMA'] = talib.SMA(df['close'].astype(float), timeperiod=14)
            df['RSI'] = talib.RSI(df['close'].astype(float), timeperiod=14)
            macd, macdsignal, macdhist = talib.MACD(df['close'].astype(float), fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd

            # Drop rows with any NaN values
            df.dropna(inplace=True)

            if df.empty:
                print("Insufficient data after preprocessing")
                return None

            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None

    def predict_price(self, instance):
        stock_symbol = self.symbol_input.text
        time_frame_value = self.selected_time_frame_value  # Use the stored numerical value
        time_frame_type = self.selected_time_frame  # Access the selected time frame value from the attribute

        if not (stock_symbol and time_frame_value and time_frame_type):
            self.feedback_text.text = "Please select symbol and time frame."
            return

        # Fetch news articles related to the stock symbol
        news_articles = self.fetch_news(stock_symbol)
        sentiment = self.analyze_sentiment(news_articles)
        self.feedback_text.text = f"Average sentiment: {sentiment}"

        # Fetch historical data
        historical_data = self.fetch_historical_data(stock_symbol, time_frame_type, time_frame_value)

        if historical_data is None:
            self.feedback_text.text = "Failed to fetch historical data"
            return

        print("Before preprocessing:")
        print(historical_data.head())

        # Calculate technical indicators
        historical_data = self.calculate_technical_indicators(historical_data)

        if historical_data is None:
            self.feedback_text.text = "Insufficient data after preprocessing"
            # Get current price
            current_price = client.get_symbol_ticker(symbol=stock_symbol)['price']
            self.feedback_text.text += f"\nCurrent Price: {current_price}\nNot enough data for accurate prediction."
            self.feedback_text.text += "\nTry again later."

            # Estimate waiting time
            if time_frame_type == '1 Hour':
                waiting_time = 7 * int(time_frame_value)
            elif time_frame_type == '5 Hours':
                waiting_time = 7 * int(time_frame_value) * 5
            elif time_frame_type == '1 Day':
                waiting_time = 7 * int(time_frame_value)
            elif time_frame_type == '2 Days':
                waiting_time = 2 * int(time_frame_value)
            self.feedback_text.text += f"\nEstimated waiting time: {waiting_time} hours"
            return

        print("After calculating technical indicators:")
        print(historical_data.head())

        # Preprocess data to handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(historical_data[['SMA', 'RSI', 'MACD']])

        print("After imputation:")
        print(X)

        # Print shape of X
        print("Shape of X:", X.shape)

        y = historical_data['close']

        print("Target variable:")
        print(y)

        # Create and fit model
        model = LinearRegression()
        model.fit(X, y)

        # Predict price
        prediction = model.predict(X)

        self.feedback_text.text = f"Predicted price: {prediction[-1]}"

if __name__ == '__main__':
    StockPricePredictor().run()
