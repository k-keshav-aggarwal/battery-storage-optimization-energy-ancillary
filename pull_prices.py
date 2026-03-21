import gridstatus
import pandas as pd

from params import nodes, start_date, end_date

iso = gridstatus.CAISO()


def get_lmp_data(start, end, nodes):
    lmp = iso.get_lmp(
        start=start,
        end=end,
        market="DAY_AHEAD_HOURLY",
        locations=nodes,
        sleep=3,
    )

    lmp_final = lmp.copy()
    lmp_final["Time"] = pd.to_datetime(lmp_final["Time"], utc=True)
    lmp_final["datetime"] = lmp_final["Time"].dt.tz_convert(None)
    lmp_final.drop(
        columns=[
            "Energy",
            "Congestion",
            "Loss",
            "Location",
            "Time",
            "Interval Start",
            "Interval End",
            "Market",
            "Location Type",
        ],
        inplace=True,
        errors="ignore",
    )
    lmp_final.rename(columns={"LMP": "SP15"}, inplace=True)
    return lmp_final


def get_as_prices_data(start_date, end_date):
    as_prices = iso.get_as_prices(date=start_date, end=end_date)

    as_final = as_prices.copy()
    as_final["Time"] = pd.to_datetime(as_final["Time"], utc=True)
    as_final["datetime"] = as_final["Time"].dt.tz_convert(None)
    as_final.drop(columns=["Time", "Interval Start", "Interval End"], inplace=True, errors="ignore")

    as_final = as_final.groupby(["datetime"]).agg({
        "Non-Spinning Reserves": "sum",
        "Regulation Down": "sum",
        "Regulation Mileage Down": "sum",
        "Regulation Mileage Up": "sum",
        "Regulation Up": "sum",
        "Spinning Reserves": "sum",
    }).reset_index()

    as_final.rename(columns={
        "Non-Spinning Reserves": "NonSpin",
        "Regulation Down": "RegDown",
        "Regulation Mileage Down": "RegDownMileage",
        "Regulation Mileage Up": "RegUpMileage",
        "Regulation Up": "RegUp",
        "Spinning Reserves": "Spin",
    }, inplace=True)

    return as_final


def get_merged_data(start_date, end_date, nodes):
    lmp_final = get_lmp_data(start_date, end_date, nodes)
    as_final = get_as_prices_data(start_date, end_date)

    merged_df = pd.merge(lmp_final, as_final, on="datetime", how="inner")

    merged_df["datetime"] = pd.to_datetime(merged_df["datetime"])
    merged_df["datetime"] = merged_df["datetime"].dt.tz_localize("UTC")
    merged_df["datetime"] = merged_df["datetime"].dt.tz_convert("US/Pacific")
    merged_df["datetime"] = merged_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged_df = merged_df.sort_values(by="datetime", inplace=False).reset_index(drop=True)

    return merged_df


def inject_price_spike(df, col="SP15", t=3, magnitude=50.0):
    out = df.copy()
    if t < 0 or t >= len(out):
        raise IndexError(f"Spike index {t} out of range for dataframe length {len(out)}")
    out.loc[t, col] = float(out.loc[t, col]) + float(magnitude)
    return out


def zscore_filter(series, threshold=2.0):
    s = series.copy()
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return s
    z = (s - mean) / std
    s.loc[z.abs() > threshold] = mean
    return s


def median_filter(series, window=5):
    s = series.copy()
    med = s.rolling(window=window, center=True, min_periods=1).median()
    return med.fillna(s)


def iqr_filter(series):
    s = series.copy()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return s.clip(lower=lower, upper=upper)


base_df = get_merged_data(start_date, end_date, nodes)

merged_df_clean = base_df.copy()
merged_df_attack = inject_price_spike(merged_df_clean, col="SP15", t=3, magnitude=50.0)

merged_df_zscore = merged_df_attack.copy()
merged_df_zscore["SP15"] = zscore_filter(merged_df_zscore["SP15"], threshold=2.0)

merged_df_median = merged_df_attack.copy()
merged_df_median["SP15"] = median_filter(merged_df_median["SP15"], window=5)

merged_df_iqr = merged_df_attack.copy()
merged_df_iqr["SP15"] = iqr_filter(merged_df_iqr["SP15"])