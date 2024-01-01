from datetime import datetime
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_boxplots_from_samples(
        df1: pd.DataFrame, df2: pd.DataFrame, asset_symbol: str = "AAPL", store: bool = True
    ) -> None:
    column_name = [
        col for col in df1.columns 
        if col.startswith(asset_symbol) 
        and col.split('_')[-2] == 'returns'
    ]
    samples = df1[column_name].values.flatten()
    ref_samples = df2[column_name].values.flatten()
    max_len = len(ref_samples)
    samples = np.pad(samples, (0, max_len - len(samples)), mode='constant', constant_values=np.nan)

    df = pd.DataFrame({
        f'{asset_symbol}_returns': samples,
        f'{asset_symbol}_returns_reference': ref_samples
    })

    # Create boxplots using Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.title(f'Boxplots for {asset_symbol} Returns and Reference')
    plt.xlabel('Sample Type')
    plt.ylabel('Returns')
    
    # Show the plot
    plt.show()

    if store:
        date = datetime.today()
        fn = f"{column_name}_{date}_boxplot.png"
        plt.savefig(os.path.join("models", "downprojection", fn))
        

def plot_boxplots_from_statistics(df: pd.DataFrame, asset_symbol: str = "AAPL", store: bool = True) -> None:
    asset_row = df[df.index.str.startswith(asset_symbol)]
    if asset_row.empty:
        print(f"Asset {asset_symbol} not found in the DataFrame.")
        return

    data_cols = ["min", "q1", "median", "mean", "q3", "max", "n"]
    ref_cols = ["ref_min_val", "ref_q1", "ref_median_val", "ref_mean_val", "ref_q3", "ref_max_val", "ref_n"]

    asset_stats = asset_row[data_cols].values.flatten()
    ref_stats = asset_row[ref_cols].values.flatten()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    axes[0].boxplot(asset_stats, labels=data_cols) # bullshit, bc it plots the values as if they were samples
    axes[0].set_title(f"{asset_symbol} - Asset Values")

    axes[1].boxplot(ref_stats, labels=ref_cols)
    axes[1].set_title(f"{asset_symbol} - Reference Values")

    p_value = asset_row["p_value"].values[0]
    fig.suptitle(f"{asset_symbol} | P-Value: {p_value}", y=1.05)

    if store:
        date = datetime.today()
        fn = f"{asset_row}_{date}_boxplot.png"
        fig.savefig(os.path.join("models", "downprojection", fn))
        
    plt.show()

