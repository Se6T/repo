import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def load_yf(ticker: str, freq: str) -> pd.DataFrame:
    tick = yf.Ticker(ticker)

    end_date = datetime.now()
    max_days = {
        '1m': 7, 
        '5m': 60, 
        '60m': 730, 
        '1d': 365*99
    }  # ideally this is stored in some config file
    start_date = end_date - timedelta(days=max_days[freq])
    df = tick.history(start=start_date, end=end_date, interval=freq)

    return df


def add_to_existing_file(df: pd.DataFrame, ticker: str, freq: str) -> None:
    """
    Find out if symbol exists in set frequency and if true, add new data to it.
    """
    raw_path = '../../data/raw'
    fn = f'{ticker}_{freq}.csv'
    file_path = os.path.join(raw_path, fn)

    if os.path.exists(file_path):
        # Load the existing CSV file
        existing_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        # Concatenate the existing DataFrame with the new data
        updated_df = pd.concat([existing_df, df], axis=0)
        # Remove duplicate indices, keeping the last occurrence
        df = updated_df[~updated_df.index.duplicated(keep='last')]

    return df


def load_asset(asset: dict, freq: str = '1d') -> None:
    ticker = asset['ticker']

    df = load_yf(ticker, freq)
    df = add_to_existing_file(df, ticker, freq)

    raw_path = os.path.join(os.getcwd(), 'data/raw')
    fn = f'{ticker}_{freq}.csv'
    file_path = os.path.join(raw_path, fn)

    df.to_csv(file_path)
