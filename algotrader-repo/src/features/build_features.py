import re
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# do all preprocessing here, including removing nans, adding features, yoy change and scaling
def remove_nans(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # get rid of nan columns
    nan_columns = set(df.columns[df.isna().all()])
    # inf_columns = set(df.columns[df.applymap(np.isinf).any()])
    cols_without_info = set([col for col in df.columns if df[col].sum() == 0.0])
    # remove_columns = list(nan_columns.union(inf_columns).union(cols_without_info))
    remove_columns = list(nan_columns.union(cols_without_info))

    df.drop(columns=remove_columns, inplace=True)
    df.bfill(inplace=True)
    df.fillna(0.0, inplace=True)
    
    if verbose and remove_columns is not None:
        print(f"Removed Columns: \n{remove_columns}")
    
    # if df.isna().sum().sum():
    #     raise ValueError(f"someting went wrong with replacing NaN s of the dataframe \n{df.columns} \n{df}")
        
    return df

def yoy_pct_change_df(
    data, 
    columns_to_keep=[
        "US-ISM-PMI_Close", 
        "weekday", 
        "day_of_month", 
        "month_in_year", 
        "low_to_low_cycle_progression", 
        "high_to_high_cycle_progression", 
        "cycle_number"], 
    normalize=True
):
    """
    Calculates the year on year percentage change. 
    Columns in columns_to_keep capture the year on year change already, hence they should remain unchanged.
    For plotting, normalized values are more informative.
    """
    df = data.copy()
    
    columns_to_keep += [col for col in df.columns if col.endswith('Volume') or re.search(r'returns_\d+d$', col)]

    for col in df.columns:
        if col not in columns_to_keep:
            try:
                df[col] = df[col].pct_change(periods=365) * 100
            except TypeError as e:
                print(f"col: {col}")
                raise(e)
            
    if normalize:
        cols_to_change = [
            col for col in df.columns 
            if col not in columns_to_keep
        ]

        if "US-ISM-PMI_Close" in df.columns:
            cols_to_change.append("US-ISM-PMI_Close")

        df.replace([np.inf, -np.inf], 0.0, inplace=True)
        df.bfill(inplace=True)
        df.fillna(0.0, inplace=True)
        scaler = StandardScaler()
        df[cols_to_change] = scaler.fit_transform(df[cols_to_change])
    
    df = df.iloc[365:]  # yoy change is nan for first year
    
    return df


def add_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    iteratively add features to the dataframe. 
    first add all custom stuff, such as business cycle progression, demark indicators
    then add pandas_ta indicators
    """
    if feature_list is None:
        return df
    else:
        # split by symbol
        # pandas_ta add features
        # concat again
        # return df
        pass


def get_returns(df: pd.DataFrame, cols: list = None, diff: int = 5) -> pd.DataFrame:
    # If cols is None, get all columns that end with 'Close'
    if cols is None:
        cols = [col for col in df.columns if col.endswith('Close')]

    # Calculate the log returns for each specified column
    for col in cols:
        df[f'{col}_returns_{diff}d'] = np.log(df[col].shift(-diff) / df[col])

    return df


def bin_volume(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    vol_cols = [col for col in df.columns if col.endswith('Volume')]

    for column in vol_cols:
        col = df[column]
        bin_edges = np.linspace(start=col.min(), stop=col.max(), num=n_bins+1)
        bin_labels = pd.cut(col, bins=bin_edges, labels=False, include_lowest=True)
        normalized_labels = bin_labels / n_bins
        df[column] = normalized_labels
    
    return df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove nans
    add features
    vol bins
    yoy_change
    normalize
    """
    df = remove_nans(df)
    df = add_features(df, feature_list=None)
    df = bin_volume(df, n_bins=5)
    df = yoy_pct_change_df(df)

    if df.isna().sum().sum():
        raise ValueError(f"someting went wrong with replacing NaN s of the dataframe \n{df.columns} \n{df}")
    
    return df

