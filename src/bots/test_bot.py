# -*- coding: utf-8 -*-
import os
import gc
import sys
import logging
from time import sleep
from pathlib import Path
from pprint import pprint
import warnings

import dotenv
import numpy as np
import pandas as pd
import talib.abstract as talib
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException


def apply_strategy(dataframe, buy_params, sell_params):

    dataframe = dataframe.rename(
        columns={"Open": "open", "Close": "close", "High": "high", "Low": "low", "Volume": "volume"})

    macd = talib.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']

    # Minus Directional Indicator / Movement
    dataframe['minus_di'] = talib.MINUS_DI(dataframe)

    # RSI
    dataframe['rsi'] = talib.RSI(dataframe)

    # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
    rsi = 0.1 * (dataframe['rsi'] - 50)
    dataframe['fisher_rsi'] = (
        np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

    # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
    dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

    # Stoch fast
    stoch_fast = talib.STOCHF(dataframe)
    dataframe['fastd'] = stoch_fast['fastd']
    dataframe['fastk'] = stoch_fast['fastk']

    # SMA - Simple Moving Average
    dataframe['sma'] = talib.SMA(dataframe, timeperiod=40)

    # Apply buying strategy
    dataframe['buy'] = np.where(
        (
            (dataframe['close'] < dataframe['sma']) &
            (dataframe['volume'] > dataframe['volume'].rolling(buy_params['buy_volumeAVG']).mean() * 4) &
            (dataframe['fastd'] > dataframe['fastk']) &
            (dataframe['rsi'] > buy_params['buy_rsi']) &
            (dataframe['fastd'] > buy_params['buy_fastd']) &
            (dataframe['fisher_rsi_norma'] < buy_params['buy_fishRsiNorma'])
        ),
        True,
        False
    )

    # Apply selling strategy
    dataframe['sell'] = np.where(
        (
            (dataframe['minus_di'] > sell_params['sell_minusDI']) &
            (dataframe['rsi'] > sell_params['sell_rsi']) &
            (dataframe['macd'] < sell_params['sell_macd'])
        ),
        True,
        False
    )

    return dataframe


class TradingBot():
    """MACD EMA Strategy TradingBot"""

    def __init__(self, binance_client: Client, logger, symbol, quote_symbol, order_qty, currency):
        super(TradingBot, self).__init__()
        self.client = binance_client
        self.logger = logger
        self.symbol = symbol
        self.quote_symbol = quote_symbol
        self.order_qty = order_qty
        self.currency = currency
        self.open_pos = False
        self.buy_price = None
        self.sell_price = None
        self.tsl_threshold = 0.99
        self.tsl_price = 0

    def get_symbol_data(self, interval, start_str):
        """Fetches and formats data from the binance api

        Args:
            client (Binance API Client)
            symbol (string)
            start_str (string)

        Returns:
            pd.DataFrame: kline df
        """
        try:
            symbol_data = pd.DataFrame(self.client.get_historical_klines(
                self.symbol, interval, start_str))
        except BinanceAPIException as error:
            print(error)
            sleep(60)
            symbol_data = pd.DataFrame(self.client.get_historical_klines(
                self.symbol, interval, start_str))

        symbol_data = symbol_data.iloc[:, :6]
        symbol_data.columns = ['Time', 'Open',
                               'High', 'Low', 'Close', 'Volume']
        symbol_data = symbol_data.set_index('Time')
        symbol_data.index = pd.to_datetime(symbol_data.index, unit='ms')
        symbol_data = symbol_data.astype(float)

        return symbol_data

    def calculate_tsl_price(self, dataframe):
        """Calculates the TSL price
        """
        tsl_price = dataframe['close'].iloc[-1] * self.tsl_threshold
        if tsl_price > self.tsl_price:
            self.tsl_price = tsl_price

    def get_trade_profit(self):
        """Returns account balance
        """
        try:
            trades = client.get_my_trades(symbol=self.symbol, limit=2)
            buy_trade = trades[0]
            sell_trade = trades[1]
            profit = float(sell_trade.get('price')) - \
                float(buy_trade.get('price'))
            print("Profit: ", profit)
        except BinanceAPIException as exception:
            pprint(exception)

    def save_order_to_disk(self, order):
        """Save order to disk
        """
        # create file if it does not exist
        file_path = os.getcwd() + '/src/bots/orders.txt'
        if not os.path.isfile(file_path):
            open(file_path, "w", encoding="utf-8").close()

        with open(file_path, "a", encoding="utf-8") as file_object:
            print("append order")
            file_object.write(order)
            file_object.write("\n")
            file_object.close()

    def create_buy_order(self):
        """Creates a buy order
        """
        try:
            order = client.order_market_buy(
                symbol=self.symbol,
                quoteOrderQty=ORDER_QTY
            )
            pprint(order.get('fills'))
            self.save_order_to_disk(order)
        except BinanceAPIException as exception:
            pprint(exception)
        except BinanceOrderException as exception:
            pprint(exception)

    def create_sell_order(self):
        """Creates a sell order
        """
        try:
            qty = float(client.get_asset_balance(
                self.quote_symbol).get('free'))
            order = client.order_market_sell(
                symbol=self.symbol,
                quantity=qty
            )
            pprint(order.get('fills'))
            self.get_trade_profit()
            self.save_order_to_disk(order)
            self.tsl_price = 0 # reset tsl price
        except BinanceAPIException as exception:
            pprint(exception)
        except BinanceOrderException as exception:
            pprint(exception)

    def trade(self):
        """Executes the trading strategy
        """
        while True:
            data = self.get_symbol_data("1m", '2 hour ago')

            buy_params = {'buy_fastd': 4, 'buy_fishRsiNorma': 28,
                          'buy_rsi': 34, 'buy_volumeAVG': 176}
            sell_params = {'sell_macd': 4, 'sell_minusDI': 12, 'sell_rsi': 50}
            data = apply_strategy(data, buy_params, sell_params)

            if self.open_pos and (data['sell'].iloc[-1]):
                self.logger.info("Placing Sell Order - Strategy")
                self.open_pos = False
                self.create_sell_order()

            if self.open_pos and (data['close'].iloc[-1] < self.tsl_price):
                self.logger.info("Placing Sell Order - TSL")
                self.open_pos = False
                self.create_sell_order()

            if data['buy'].iloc[-1] and not self.open_pos:
                self.logger.info("Placing Buy Order")
                self.open_pos = True
                self.create_buy_order()
                self.calculate_tsl_price(data)

            # update tsl price if we haven an open position
            if self.open_pos:
                self.calculate_tsl_price(data)

            gc.collect()
            sleep(20)


def main(binance_client, symbol, coin, order_qty, currency):
    """
        Starts the Trading Bot
    """
    logger = logging.getLogger(__name__)
    logger.info('Start Trading Bot')
    bot = TradingBot(binance_client, logger, coin, symbol, order_qty, currency)
    bot.trade()


if __name__ == '__main__':
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

if not sys.warnoptions:
    warnings.simplefilter("ignore")

    # Parameters
    COIN = 'BTC'
    SYMBOL = 'BTCUSDT'
    CURRENCY = 'USDT'
    ORDER_QTY = 50

    USE_DEMO_ACCOUNT = True
    ROOT_DIR = Path(__file__).resolve().parents[2]

    # load env file
    dotenv_path = os.path.join(ROOT_DIR, '.env')
    dotenv.load_dotenv(dotenv_path)

    api_key = os.environ.get('BINANCE_API_KEY_TEST')
    api_secret = os.environ.get('BINANCE_SECRET_KEY_TEST')
    client = Client(api_key, api_secret, testnet=True)
    print("USE DEMO ACCOUNT!")

    main(client, COIN, SYMBOL, ORDER_QTY, CURRENCY)
