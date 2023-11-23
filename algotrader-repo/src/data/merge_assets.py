import os
import pandas as pd
from datetime import datetime


def load_data(file_path: str, index_name='Date') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    date_start_idx = 0

    # Deal with dates before 1970
    dates = df[index_name]
    for date in dates:
        if int(date.split('-')[0]) < 1970:
            date_start_idx += 1
    df = df.iloc[date_start_idx:, :]
    return df


def convert_date_index(df: pd.DataFrame, filename: str, index_name='Date') -> pd.DataFrame:
    idx = df[index_name]
    new_idx = []

    for el in idx:
        str_format = "%Y-%m-%d" if filename.startswith("fred") else "%Y-%m-%d %H:%M:%S%z"
        dt = datetime.strptime(el, str_format)

        try:
            timestamp = int(dt.timestamp())
        except Exception as e:
            print(f"dt at el {el}: \n{dt}")
            raise e

        date_string = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
        new_idx.append(date_string)

    df['Date'] = pd.to_datetime(new_idx)
    df.set_index('Date', inplace=True)
    return df


def rename_columns(df: pd.DataFrame, ticker: str, index_name='Date') -> pd.DataFrame:
    df = df.rename(columns={col: f"{ticker}_" + col for col in df.columns if col != index_name})
    return df


def drop_columns(df: pd.DataFrame, index_name='Date') -> pd.DataFrame:
    drop_cols = [
        col for col in df.columns 
        if col.endswith('Splits') 
        or col.endswith('Dividends')
        or col.endswith('Gains')
        or col.endswith(index_name)
    ]
    df.drop(columns=drop_cols, inplace=True)
    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index
    dates = pd.to_datetime(dates)
    weekdays = [date.weekday() / 6.0 for date in dates]
    days_of_month = [(date.day - 1) / (date.days_in_month - 1) for date in dates]
    month_in_year = [(date.month - 1) / 11.0 for date in dates]

    df['weekday'] = weekdays
    df['day_of_month'] = days_of_month
    df['month_in_year'] = month_in_year
    return df


def merge_assets(output_filepath: str, freq: str, index_name='Date') -> None:
    raw_path = '../../data/raw'
    file_list = os.listdir(raw_path)
    full_df = pd.DataFrame()

    for filename in file_list:
        file_path = os.path.join(raw_path, filename)
        df = load_data(file_path, index_name)

        ticker = filename.split('_')[0]
        if ticker.startswith("fred"):
            ticker = filename.split('_')[1][:-4]
            df.columns = ["Date", "Close"]

        df = convert_date_index(df, index_name)
        df = rename_columns(df, ticker, index_name)

        full_df = pd.concat([full_df, df], axis=1, join='outer')

    full_df = drop_columns(full_df, index_name)
    full_df = full_df.sort_index()
    full_df = add_datetime_features(full_df)

    fn = f"merged_assets_{datetime.today().date()}.csv"
    full_df.to_csv(os.path.join(output_filepath, fn))
