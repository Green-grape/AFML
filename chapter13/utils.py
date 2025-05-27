import numpy as np
import pandas as pd


def get_dollar_bars(df: pd.DataFrame, dollar_threshold: float):
    dollar = df["Volume"] * df["Close"]
    dollar_sum = 0
    start_idx = 0
    dollar_bars = pd.DataFrame()
    for i in range(len(df)):
        dollar_sum += dollar.iloc[i]
        if dollar_sum >= dollar_threshold:
            bar = {
                "Open": df.iloc[start_idx]["Open"],
                "High": df.iloc[start_idx : i + 1]["High"].max(),
                "Low": df.iloc[start_idx : i + 1]["Low"].min(),
                "Close": df.iloc[i]["Close"],
                "Volume": df.iloc[start_idx : i + 1]["Volume"].sum(),
            }
            if dollar_bars.empty:
                dollar_bars = pd.DataFrame(bar, index=[df.index[i]])
            else:
                dollar_bars = pd.concat(
                    [dollar_bars, pd.DataFrame(bar, index=[df.index[i]])]
                )
            start_idx = i + 1
            dollar_sum = 0
    if dollar_sum > 0:
        bar = {
            "Open": df.iloc[start_idx]["Open"],
            "High": df.iloc[start_idx:]["High"].max(),
            "Low": df.iloc[start_idx:]["Low"].min(),
            "Close": df.iloc[-1]["Close"],
            "Volume": df.iloc[start_idx:]["Volume"].sum(),
        }
        dollar_bars = pd.concat([dollar_bars, pd.DataFrame(bar, index=[df.index[i]])])
    return dollar_bars
    return dollar_bars
