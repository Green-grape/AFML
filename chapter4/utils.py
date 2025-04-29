"""
Before chapters function for exercises
"""

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


def get_ema_daily_volatility(close: pd.Series, span: int = 100):
    """
    daily return을 계산하여 일일 변동성을 추정하는 함수
    """
    daily_index = pd.DatetimeIndex(
        sorted({pd.Timestamp(t.date()) for t in close.index})
    )

    def get_before_day(t: pd.Timestamp):
        date_to_daily = pd.Timestamp(t.date())
        before_day = daily_index[daily_index < date_to_daily]
        if len(before_day) == 0:
            return t - pd.Timedelta(days=1)
        else:
            date_diff = date_to_daily - before_day[-1]
            return t - date_diff

    ret = close.index.searchsorted([get_before_day(t) for t in close.index])
    ret = ret[ret > 0]
    ret = pd.Series(close.index[ret], index=close.index[-len(ret) :])
    daily_ret = close.loc[ret.index] / close.loc[ret.values].values - 1
    daily_vol = daily_ret.ewm(span=span).std()
    return daily_vol.dropna()


def apply_cumsum_filter(df: pd.DataFrame, filter_s: pd.Series):
    diff = df["Close"].pct_change().dropna().diff().dropna()
    intersect_idx = diff.index.intersection(filter_s.index)

    # index 통일
    diff = diff.loc[intersect_idx]
    filter_s = filter_s.loc[intersect_idx]
    df = df.loc[intersect_idx]

    s_pos, s_neg = 0, 0
    start_idx = 0
    cumsum_bar = pd.DataFrame()

    for i in range(len(diff)):
        s_pos = max(0, s_pos + diff.iloc[i])
        s_neg = max(0, s_neg + diff.iloc[i])
        s = max(s_pos, -s_neg)
        if s > filter_s.iloc[i]:
            sub_df = df.iloc[start_idx : i + 1]
            bar = {
                "Open": sub_df.iloc[0]["Open"],
                "High": sub_df["High"].max(),
                "Low": sub_df["Low"].min(),
                "Close": sub_df.iloc[-1]["Close"],
                "Volume": sub_df["Volume"].sum(),
            }
            if cumsum_bar.empty:
                cumsum_bar = pd.DataFrame(bar, index=[df.index[i]])
            else:
                cumsum_bar = pd.concat(
                    [cumsum_bar, pd.DataFrame(bar, index=[df.index[i]])]
                )
            start_idx = i + 1
            s_pos, s_neg = 0, 0
    return cumsum_bar


def apply_triple_barrier_method(
    close: pd.Series, events: pd.Series, barrier_width: tuple
) -> pd.DataFrame:
    ret = events[["t1"]].copy(deep=True)
    upper_barrier_width, lower_barrier_width = barrier_width
    if upper_barrier_width > 0:
        profit_taking = events["trgt"] * upper_barrier_width
    else:
        profit_taking = pd.Series(index=events.index, data=[np.nan] * len(events))
    if lower_barrier_width > 0:
        stop_loss = -events["trgt"] * lower_barrier_width
    else:
        stop_loss = pd.Series(index=events.index, data=[np.nan] * len(events))
    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        path_price = close.loc[loc:t1]
        if len(path_price) == 0:
            ret.loc[loc, "pt"] = np.nan
            ret.loc[loc, "sl"] = np.nan
            continue
        path_return = (path_price / path_price.iloc[0]) - 1
        ret.loc[loc, "pt"] = path_return[
            path_return > profit_taking.loc[loc]
        ].index.min()
        ret.loc[loc, "sl"] = path_return[path_return < stop_loss.loc[loc]].index.min()
    return ret


def get_vertical_barrier(
    close: pd.Series, t_events: pd.DatetimeIndex, num_days: int = 1
) -> pd.Series:
    daily_index = pd.DatetimeIndex(
        sorted({pd.Timestamp(t.date()) for t in close.index})
    )

    def get_after_day(t: pd.Timestamp):
        date_to_daily = pd.Timestamp(t.date()) + pd.Timedelta(days=num_days)
        after_day = daily_index[daily_index > date_to_daily]
        if len(after_day) == 0:
            return t + pd.Timedelta(days=num_days)
        else:
            date_diff = after_day[0] - date_to_daily
            return t + date_diff

    t1 = close.index.searchsorted([get_after_day(t) for t in t_events])
    t1 = t1[t1 < len(close)]
    t1 = pd.Series(data=close.index[t1], index=t_events[: len(t1)])
    return t1


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    barrier_width: tuple,
    target: pd.Series,
    min_ret: float = 0.0,
    t1: pd.Series = None,
):
    target = target.loc[t_events]
    target = target[target > min_ret]

    if t1 is None:
        t1 = pd.Series(pd.NaT, index=t_events)

    side_ = pd.Series(1.0, index=target.index)
    events = pd.concat({"t1": t1, "trgt": target, "side": side_}, axis=1).dropna(
        subset=["trgt"]
    )

    triple_barrier = apply_triple_barrier_method(close, events, barrier_width)
    events["t1"] = triple_barrier.dropna(how="all").min(axis=1)
    events = events.drop("side", axis=1)
    return events


def get_bins(events: pd.DataFrame, close: pd.Series, t1: pd.Series):
    """
    Get the bins for the events.

    :param events: pd.DataFrame The events dataframe containing the event times and labels.
    close: pd.Series The close price series.
    t1: pd.Series
        The vertical barrier time series.
    :return: pd.DataFrame
        A dataframe with the event times and the corresponding bins.
        (0: touch vertical barrier, 1: profit taking, -1: stop loss)
    """
    # check vertical barrier touch first
    vt_time_stamp = pd.DatetimeIndex([])
    for loc, touch_time in events["t1"].dropna().items():
        if (loc in t1.index) and t1.loc[loc] == touch_time:
            vt_time_stamp = vt_time_stamp.append(pd.DatetimeIndex([loc]))

    # check event time
    events_ = events.dropna(subset=["t1"])
    price_idx = events_.index.union(events_["t1"].values).drop_duplicates()
    price = close.reindex(price_idx, method="bfill")
    ret = pd.DataFrame(index=events_.index)
    ret["ret"] = price.loc[events_["t1"]].values / price.loc[events_.index].values - 1
    ret["bin"] = np.sign(ret["ret"])
    ret.loc[vt_time_stamp, "bin"] = 0
    return ret
