# -*- coding: utf-8 -*-
import logging
import yfinance as yf
from pathlib import Path


class DownloadAndCreateDataset():
    """Download and create dataset

    Args:
        tickers (List): List of tickers to download
        start_date (String): Start date of download
        end_date (String): End date of download
        interval (String): Interval
        path (String): Path to save csv files

    Example:
        tickers = ['BA', 'UNH', 'MCD', 'HD'],
        start_date = '2019-01-01',
        end_date = '2021-06-01',
        interval = '1d',
        path = './'
    """

    def __init__(self, tickers, start_date, end_date, interval, path='./'):
        super(DownloadAndCreateDataset, self).__init__()
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.path = path

    def create_dataset(self):
        """Download data from yFinance and store it as a csv file
        """
        for ticker in self.tickers:
            data = yf.download(
                tickers=ticker,
                start=self.start_date,
                end=self.end_date,
                interval='1d',
                actions=False,
                prepost=True,
                auto_adjust=True,
                back_adjust=True,
                index_as_date=True,
                threads=True,
            )
            data.to_csv(self.path + "/" + ticker + ".csv")


def main():
    """
        Downloads the data and stores it in a sql file under /data
    """
    logger = logging.getLogger(__name__)
    logger.info('start downloading data')

    dataset_service = DownloadAndCreateDataset(
        tickers=[
            # Indexes & ETFs
            '^GSPC',    # S&P 500
            '^GDAXI',   # DAX
            'XWD.TO',   # iShares MSCI World Index ETF

            # Stocks
            'AAPL',     # Apple
            'MSFT',     # Microsoft
            'GOOG',     # Google
            'AMZN',     # Amazon
            'NVDA',     # NVIDIA
            'ORCL',     # Oracle
            'NFLX',     # Netflix
            'SAP',      # SAP
            'PYPL',     # PayPal
            'ABNB',     # AirBnB
            'BKNG',     # Booking
            'SHOP',     # Shopify
            'UBER',     # Uber
            'SNOW',     # Snowflake
            'VMW',      # Vmware
            'CRWD',     # Crowstrike
            'COIN',     # Coinbase
            'DDOG',     # Datadog
            'EA',       # EA
            'NET',      # Cloudflare
            'CRM',      # Salesforce
            'BNTX',     # Biontec

            # Transparency Index
            'TER',      # TERADYNE INC
            'COST',     # COSTCO WHOLESALE CORP
            'IDCC',     # INTERDIGITAL INC
            'WDAY',     # WORKDAY INC
            'SPOT',     # SPOTIFY TECHNOLOGY SA
            'GMAB',     # GENMAB A/S
            'DOCN',     # DIGITALOCEAN HOLDINGS INC
            'FIVN',     # FIVE9 INC
            'AXON',     # AXON ENTERPRISE INC
            'GRMN',     # GARMIN LTD
            'BILL',     # BILL.COM HOLDINGS INC
            'HUBS',     # HUBSPOT INC
            'SPLK',     # SPLUNK INC
            'ZM',       # ZOOM VIDEO COMMUNICATIONS INC
            'DOCU',     # DOCUSIGN INC

            # Coins
            'BTC-EUR',  # Bitcoin
            'ETH-USD',  # Ethereum
        ],

        start_date='2015-01-01',
        end_date='2022-03-31',
        interval='1d',
        path="/workspaces/Stock-Trading/data"
    )

    dataset_service.create_dataset()

    logger.info('done downloading data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
