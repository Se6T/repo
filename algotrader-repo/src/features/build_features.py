import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# do all preprocessing here, including removing nans, adding features, yoy change and scaling
def remove_nans(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # get rid of nan columns
    nan_columns = set(df.columns[df.isna().all()])
    inf_columns = set(df.columns[df.isinf().all()])
    cols_without_info = set([col for col in df.columns if df[col].sum() == 0.0])
    remove_columns = list(nan_columns.union(inf_columns).union(cols_without_info))

    df.drop(columns=remove_columns, inplace=True)
    df.ffill(inplace=True)

    
    if verbose and remove_columns is not None:
        print(f"Removed Columns: \n{remove_columns}")
    
    if df.isna().sum().sum() or df.isinf().sum().sum():
        raise ValueError(f"someting went wrong with replacing NaN s of the dataframe \n{df.columns} \n{df}")
        
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
    normalize=False
):
    """
    Calculates the year on year percentage change. 
    Columns in columns_to_keep capture the year on year change already, hence they should remain unchanged.
    For plotting, normalized values are more informative.
    """
    df = data.copy()
    
    for col in df.columns:
        if col not in columns_to_keep:
            df[col] = df[col].pct_change(periods=365) * 100
    
    if normalize:
        cols_to_change = [col for col in df.columns if col not in columns_to_keep]
        cols_to_change.append("US-ISM-PMI_Close")
        scaler = StandardScaler()
        df[cols_to_change] = scaler.fit_transform(df[cols_to_change])
    
    df = df[df['cycle_number'] > 0]  # only take full business cycles
    
    return df



def add_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    iteratively add features to the dataframe. 
    first add all custom stuff, such as business cycle progression, demark indicators
    then add pandas_ta indicators
    """

    pass


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    pass

