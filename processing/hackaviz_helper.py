#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File name: hackaviz_helper.py
# Author: XSS J16271-2423
# Date created: 2025-03-15
# Version = "1.0"
# License =  "MIT license"
# =============================================================================
""" Helper functions for the DataSwag Hackaviz exploratory data analysis notebook"""
# =============================================================================


import os
import warnings
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from joypy import joyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

from numpy import ndarray
from typing import Dict, List, Tuple, Optional, Union, Sequence


# ---------------------------------------------------------------------------
# Parameters and setup

# Type aliases
DataFrame = pd.DataFrame
NDArray = np.ndarray

# Display
pd.set_option("display.float_format", lambda x: "%.2f" % x)

plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams["axes.formatter.use_mathtext"] = False
plt.rcParams["axes.formatter.limits"] = (-7, 7)

# Constants
DATASET_URLS = {
    "hauteur_eau_9_crues": "https://github.com/Toulouse-Dataviz/hackaviz-2025/blob/main/data/hauteur_eau_9_crues.parquet",
    "hauteur_eau_serie_longue_toulouse": "https://github.com/Toulouse-Dataviz/hackaviz-2025/raw/main/data/hauteur_eau_serie_longue_toulouse.parquet"
}

FIG_SIZES = {
    "single": (8, 6),
    "missing": (10, 8),
    "grid": (12, 10),
    "joyplot": (6, 7),
    "lineplots": (10, 4),
    "animate": (10, 6)
}

Z_THRESH = 4.1
MIN_STD = 1e-6

DAY_RANGE = np.arange(1, 367)
MONTH_TICKS = {
    "positions": [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
    "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
}

STYLES = {
    "classic": {
        "title": "167 YEARS OF DATA",
        "years": [],
        "cmap": None,
        "norm_func": None,
        "linewidth": 0.6,
        "highlight_width": None,
        "bgcolor": "k",
        "default_color": "w",
        "flood_metric": lambda d: d.groupby("Year")["hauteur"].max()
    },
    "calm": {
        "title": "C A L M E",
        "years": ["2017", "1989", "1967", "1983", "1997", "1946", "2023", "2012", "2016"],
        "cmap": plt.get_cmap("winter"),
        "norm_func": lambda mn, mx: mcolors.Normalize(vmin=mn, vmax=mx),
        "linewidth": 0.6,
        "highlight_width": 1.3,
        "bgcolor": "grey",
        "default_color": "bisque",
        "flood_metric": lambda d: (
            d.groupby("Year")["hauteur"]
            .apply(lambda x: x[x != 0].std())
        )
    },
    "excess": {
        "title": "E X C È S",
        "years": ["1857", "1875", "1879", "1900", "1905", "1952", "1977", "2000", "2022"],
        "cmap": plt.get_cmap("gist_heat_r"),
        "norm_func": lambda vmin, vmax: mcolors.PowerNorm(gamma=1.5,
                                                          vmin=vmin,
                                                          vmax=vmax),
        "linewidth": 0.6,
        "highlight_width": 1.3,
        "bgcolor": "grey",
        "default_color": "bisque",
        "flood_metric": lambda daily: (
            daily.groupby("Year")["hauteur"]
            .apply(lambda x: x[x != 0].max())
        )
    },
    "occitania": {
        "title": "O C C I T A N I A",
        "years": ["1857", "1875", "1879", "1900", "1905", "1952", "1977", "2000", "2022"],
        "cmap": "gold",
        "norm_func": None,
        "linewidth": 0.6,
        "highlight_width": 1.3,
        "bgcolor": "crimson",
        "default_color": "bisque",
        "flood_metric": lambda daily: (
            daily.groupby("Year")["hauteur"]
            .apply(lambda x: x[x != 0].max()))
    }
}


# ---------------------------------------------------------------------------
# Utility functions

def download_file(url: str, dest: str) -> None:
    """
    Download a file from a given URL to a specified destination.

    Args:
        url (str): The URL of the file to download.
        dest (str): The local file path where the downloaded file will be saved.

    Returns:
        None. Saves the file to the specified destination.

    Raises:
        requests.exceptions.RequestException: If there is an error during download.
    """
    print("Downloading..")
    resp = requests.get(url)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    print(f"Downloaded: {dest}")


def load_flood_dataset(
    folder: str,
    delimiter: str
) -> Optional[Tuple[DataFrame, str]]:
    """
    Load a parquet dataset matching a specific delimiter or download if not found.
    The specified folder will be created if it does not already exists.

    Args:
        folder (str): Directory to search for or download datasets.
        delimiter (str): Specific identifier keyword to match in the dataset filename.

    Returns:
        Optional[Tuple[DataFrame, str]]: A tuple containing the loaded DataFrame 
        and its file path, or None if no matching dataset is found.
    """
    os.makedirs(folder, exist_ok=True)
    files = [fn for fn in os.listdir(folder) if fn.endswith(
        ".parquet") and delimiter in fn]

    if not files:
        for name, url in DATASET_URLS.items():
            dest = os.path.join(folder, f"{name}.parquet")
            if not os.path.exists(dest):
                download_file(url, dest)
        files = [fn for fn in os.listdir(folder) if fn.endswith(
            ".parquet") and delimiter in fn]

    if not files:
        print(f"No dataset with '{delimiter}' found after download.")
        return None

    path = os.path.join(folder, files[0])
    df = pd.read_parquet(path, engine="pyarrow")
    print("Loaded:", df.shape)
    return df, path


def find_missing_periods(
    df: DataFrame,
    dt_col: str = "date_heure"
) -> Dict[str, List[str]]:
    """
    Identify missing years, months, and days in a time series (datetime column).

    Args:
        df (DataFrame): Input DataFrame to analyze.
        dt_col (str, optional): Name of the datetime column. Defaults to "date_heure".

    Returns:
        Dict[str, List[str]]: A dictionary with lists of missing years, months, and days.
    """
    dates = pd.to_datetime(df[dt_col]).dt.tz_convert(
        "UTC").dt.tz_localize(None)
    tmin, tmax = dates.min(), dates.max()
    yrs = np.arange(tmin.year, tmax.year+1)
    months = pd.period_range(tmin.to_period(
        "M"), tmax.to_period("M"), freq="M")
    days = pd.date_range(tmin, tmax, freq="D")

    missing_years = [str(y) for y in yrs if y not in dates.dt.year.unique()]
    present_months = dates.dt.to_period("M").unique()
    missing_months = [p.strftime("%m-%Y")
                      for p in months.difference(present_months)]
    present_days = dates.dt.normalize().unique()
    missing_days = [d.strftime("%Y-%m-%d")
                    for d in days.normalize().difference(present_days)]

    return {"missing_years": missing_years, "missing_months": missing_months, "missing_days": missing_days}


def detect_outliers(
    df: DataFrame,
    dt_col: str = "date_heure",
    water_lev_col: str = "hauteur",
    z_thresh: float = Z_THRESH,
    min_std: float = MIN_STD
) -> DataFrame:
    """
    Detect outliers in a DataFrame using group-wise Z-score thresholding.

    This function identifies outliers by:
    1. Grouping data by station code and date
    2. Calculating group-wise statistics
    3. Computing Z-scores
    4. Filtering based on Z-score threshold and standard deviation

    Args:
        df (DataFrame): Input DataFrame containing the data.
        dt_col (str, optional): Name of the datetime column. Defaults to "date_heure".
        water_lev_col (str, optional): Name of the column with numerical values. Defaults to "hauteur".
        z_thresh (float, optional): Z-score threshold for outlier detection. Defaults to Z_THRESH.
        min_std (float, optional): Minimum standard deviation for valid groups. Defaults to MIN_STD.

    Returns:
        DataFrame: A DataFrame containing the detected outliers.
    """
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy[dt_col]).dt.date
    stats_df = df_copy.groupby(["code_station", "date"])[water_lev_col].agg(
        n="size", mean="mean", std="std").reset_index()
    merged = df_copy.merge(stats_df, on=["code_station", "date"])
    merged["z_score"] = (merged[water_lev_col] - merged["mean"]
                         ) / merged["std"].replace(0, np.nan)
    mask = (merged["n"] > 1) & (merged["std"] >= min_std) & (
        merged["z_score"].abs() > z_thresh)
    outliers = merged[mask]
    print(
        f'{outliers["date"].nunique()} days containing {len(outliers)} outliers')

    return outliers


def prepare_data(
    df: DataFrame,
    dt_col: str = "date_heure",
    water_lev_col: str = "hauteur",
) -> Tuple[ndarray, List[str], DataFrame]:
    """
    Prepare time series data for analysis by:
    1. Parsing datetime column to midnight timestamps
    2. Creating a full day index from Jan 1 of min year to Dec 31 of max year
    3. Aggregating values by day (keeping maximas)
    4. Merging into full index, keeping NaNs for missing days
    5. Pivoting into (years x days) array

    Args:
        df (DataFrame): Input DataFrame containing time series data.
        dt_col (str, optional): Name of the datetime column. Defaults to "date_heure".
        water_lev_col (str, optional): Name of the column with numerical values. Defaults to "hauteur".

    Returns:
        Tuple containing:
        - ndarray: 2D array of daily maximum values (years x days)
        - List[str]: List of year labels
        - DataFrame: Processed daily data
    """
    # 1) copy & normalize to midnight
    df_copy = df.copy()
    df_copy[dt_col] = pd.to_datetime(df_copy[dt_col]).dt.normalize()
    first_year = df_copy[dt_col].dt.year.min()     # extract year span
    last_year = df_copy[dt_col].dt.year.max()
    # 2) full daily index from Jan 1 first_year to Dec 31 last_year
    full = pd.DataFrame({
        dt_col: pd.date_range(
            start=pd.Timestamp(first_year, 1, 1),
            end=pd.Timestamp(last_year, 12, 31),
            freq="D",
        )
    })
    # 3) daily max of water_lev_col
    daily_max = (
        df_copy
        .groupby(dt_col, as_index=False)[water_lev_col]
        .max()
    )
    daily_max[dt_col] = daily_max[dt_col].dt.tz_localize(None)
    # 4) merge & keep NaNs
    daily_metrics_df = (
        full
        .merge(daily_max, on=dt_col, how="left")
        .assign(
            # hauteur will now contain NaN wherever no original data existed
            hauteur=lambda d: d[water_lev_col].fillna(0),
            # missing values flag - True if exactly 0
            is_zero=lambda d: d[water_lev_col] == 0,
            Year=lambda d: d[dt_col].dt.year,
            Day=lambda d: d[dt_col].dt.dayofyear,
        )
    )
    # 5) pivot: years × day-of-year, leave NaNs in pivot
    pivot = (
        daily_metrics_df
        .pivot(index="Year", columns="Day", values=water_lev_col)
        .reindex(columns=DAY_RANGE)     # ensure columns 1…366
        .sort_index(ascending=False)    # newest year first
    )
    year_day_matrix = pivot.to_numpy()     # shape (n_years, 366), with NaNs
    yr_labels = pivot.index.astype(str).tolist()

    return year_day_matrix, yr_labels, daily_metrics_df


# ---------------------------------------------------------------------------
# Plot functions

def plot_frequency_histogram(df: DataFrame) -> None:
    """
    Plot histograms for each numeric column in the DataFrame.

    Args:
        df (DataFrame): Input DataFrame with numeric columns to visualize.

    Returns:
        None. Displays histogram plots for each numeric column.
    """
    num_cols = df.columns
    fig, axes = plt.subplots(1, len(num_cols), figsize=(
        6 * len(num_cols), 6), squeeze=False)
    for ax, col in zip(axes[0], num_cols):
        df[col].hist(ax=ax)
        ax.set_title(f"Frequ. distribution in {col}")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.grid(False, axis="x")
    plt.show()


def plot_missing_values(df: DataFrame, dataset_filepath: str) -> None:
    """
    Analyze and visualize missing values in the dataset.

    Args:
        df (DataFrame): Input DataFrame to analyze for missing values.
        dataset_filepath (str): Full path to the dataset file for reporting.

    Returns:
        None. Prints missing value statistics and displays visualization if applicable.
    """
    print(f"Missing data statistics for {dataset_filepath}:")
    print(df.isnull().sum())

    if df.isnull().values.any():
        print(f"rows with missing values: {df[df.isnull().any(axis=1)]}")
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                print(f"Column {column} has missing values.")
                print(f"Missing data pattern for {dataset_filepath}:")
                plt.figure(figsize=FIG_SIZES["missing"])
                sns.heatmap(df.isnull(), cbar=False)
                plt.title(f"Missing data pattern for {dataset_filepath}")
                plt.show()

                for other_column in df.columns:
                    if column != other_column:
                        correlation = df[column].isnull().corr(
                            df[other_column].isnull())
                        if correlation > 0.5:
                            print(
                                f"Missing data in column {column} is correlated with column {other_column}.")
                            print("Missing data is likely missing at random (MAR).")
                        else:
                            print(
                                f"Missing data in column {column} is not correlated with column {other_column}.")
                            print(
                                "Missing data is likely missing not at random (MNAR).")
    else:
        print("Plot not necessary, no missing data found.")


def plot_distribution(df: DataFrame, grid: bool = False) -> None:
    """
    Plot comprehensive distribution visualizations for numeric columns.

    Generates histograms, kernel density estimates, Q-Q plots, and boxplots 
    for each numeric column in the DataFrame.

    Args:
        df (DataFrame): Input DataFrame with numeric columns.
        grid (bool, optional): Whether to display plots in a grid layout. Defaults to False.

    Returns:
        None. Displays distribution plots and prints summary statistics.
    """
    num_cols = df.select_dtypes(include=["float", "int"]).columns
    for col in num_cols:
        if grid:
            fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["grid"])
            axes = axes.flatten()
        else:
            axes = [plt.figure(figsize=FIG_SIZES["single"]).gca()
                    for _ in range(4)]

        df[col].hist(ax=axes[0], bins=50)
        axes[0].set_title(f"Histogram: {col}")
        sns.kdeplot(df[col].dropna(), fill=True, ax=axes[1])
        axes[1].set_title(f"Density: {col}")
        stats.probplot(df[col].dropna(), plot=axes[2])
        pts, fit = axes[2].get_lines()
        pts.set_markerfacecolor("#7ca1cc")
        pts.set_markeredgecolor("#7ca1cc")
        fit.set_color("#a50b5e")
        fit.set_linestyle("--")
        axes[3].boxplot(df[col].dropna(), vert=True)
        axes[3].set_title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.show()

        summary = df[col].agg(["mean", "median", "std", "skew", "kurtosis"])
        mode = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
        print(f"{col} — mean: {summary['mean']}, med.: {summary['median']}, mode: {mode}, "
              f"std: {summary['std']}, skew.: {summary['skew']}, kurt.: {summary['kurtosis']}")


def plot_original_joyplot(df: pd.DataFrame, dt_col: str = "date_heure", water_lev_col: str = "hauteur") -> None:
    """
    Plot a classic joyplot of daily maximum water levels by year.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        dt_col (str, optional): Name of the datetime column. Defaults to "date_heure".
        water_lev_col (str, optional): Name of the column with numerical values. Defaults to "hauteur".

    Returns:
        None. Displays a joyplot of daily maximum water levels.
    """
    df_copy = df.copy()
    df_copy[dt_col] = pd.to_datetime(df_copy[dt_col])
    daily_metrics_df = (df_copy.assign(day=lambda d: d[dt_col].dt.normalize()).groupby("day", as_index=False)[
        water_lev_col].max().rename(columns={"day": "date"}).assign(Year=lambda d: d["date"].dt.year,))
    years_desc = sorted(daily_metrics_df["Year"].unique(), reverse=True)
    daily_metrics_df["Year"] = pd.Categorical(
        daily_metrics_df["Year"], categories=years_desc, ordered=True)
    vmin, vmax = daily_metrics_df[water_lev_col].min(
    ), daily_metrics_df[water_lev_col].max()
    fig, axes = joyplot(
        daily_metrics_df, by="Year", column=water_lev_col, kind="counts", bins=80, overlap=0.5,
        figsize=FIG_SIZES["joyplot"], grid=False, fill=False, background="k", linecolor="w",
        linewidth=1, legend=False, xlabels=False, ylabels=False, xlim=(vmin, vmax),)

    fig.patch.set_facecolor("k")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


def plot_byflavor(
    year_x_day_data: NDArray,
    yr_labels: List[str],
    daily_metrics_df: DataFrame,
    style_key: str = "calm"
) -> None:
    """Plot yearly data as a styled joyplot, highlighting specific years.

    This function takes a 2D array of daily measurements (years x days),
    a list of year labels, and a DataFrame of daily flood metrics. It
    applies a pre-defined style (from the global STYLES dict) to color
    each year's ridgeline according to a flood metric, masks (overpaints)
    missing "zero sentinel" days, and emphasizes flood years via line width.

    Args:
        year_x_day_data (NDArray):
            A 2D NumPy array of shape (n_years, n_days).  Each row is one
            year's daily values; zero indicates a missing data sentinel.
        yr_labels (List[str]):
            A list of length n_years giving a string label for each row in
            year_x_day_data (typically the year as a string).
        daily_metrics_df (DataFrame):
            A pandas DataFrame containing daily-aggregated flood metrics
            (e.g. daily maxima or volumes) indexed by year and day.
        style_key (str, optional):
            Key into the global STYLES mapping, which provides:
              - "years": List of years to highlight
              - "flood_metric": a callable that returns a Series of metrics
              - "norm_func": optional normalization factory
              - "cmap": color map or fixed color
              - other plot styling parameters (linewidths, bgcolor, title).
            Defaults to "calm".

    Returns:
        None: Displays the styled joyplot inline.

    Raises:
        KeyError:
            If `style_key` is not found in the global STYLES dictionary.
        ValueError:
            If `daily_metrics_df` does not contain the necessary index
            or columns expected by the chosen flood_metric.
    """
    style = STYLES[style_key]
    metric = style["flood_metric"](daily_metrics_df)
    # Filtered & ordered” version of metric data
    fv = metric.reindex(map(int, style["years"])).astype(float)
    if style["norm_func"] is not None:
        norm = style["norm_func"](fv.min(skipna=True), fv.max(skipna=True))
        highlight_cols = {
            y: style["default_color"] if np.isnan(fv.loc[int(y)])
            else style["cmap"](norm(fv.loc[int(y)]))
            for y in style["years"]
        }
    else:
        # For styles without normalization, use a single color
        highlight_cols = {y: style["cmap"] for y in style["years"]}

    colors = [highlight_cols.get(lbl, style["default_color"])
              for lbl in yr_labels]
    fig, axes = joyplot(
        year_x_day_data.tolist(), labels=yr_labels,
        xlabels=False, ylabels=False,
        x_range=DAY_RANGE,
        linewidth=style["linewidth"],
        fill=False,
        color=colors,
        kind="values",
        overlap=0.8,
        figsize=FIG_SIZES["joyplot"],
        background=style["bgcolor"]
    )

    for ax, lbl in zip(axes, yr_labels):
        line = ax.get_lines()[0]
        x_data, y_data = line.get_data()
        # /!\ mask those plotted zero flags (NA values)
        zero_points = (y_data == 0)
        ax.plot(
            x_data[zero_points],
            y_data[zero_points],
            color=style["bgcolor"],
            linewidth=style["linewidth"]*1.5,
            solid_capstyle="butt",
            antialiased=False,
            zorder=line.get_zorder()+1
        )
        # highlight if this is a highlighted year
        if str(lbl) in style["years"]:
            line.set_linewidth(style["highlight_width"])
        else:
            line.set_linewidth(style["linewidth"])

        ax.set_xlim(1, 366)
        ax.set_ylim(-500, 8600)
        ax.tick_params(colors="w")
        ax.set_facecolor(style["bgcolor"])

    fig.suptitle(style["title"], color="w", fontsize=14)
    fig.patch.set_facecolor(style["bgcolor"])
    plt.show()


def plot_years(
    year_x_day_data: np.ndarray,
    yr_labels: Sequence[str],
    years: Union[int, range, Sequence[int]],
    interpolate_gaps: bool = True,
    pad: float = 0.05
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    For each year in `years`, plot one dark-background subplot of
    daily-max values vs. day-of-year. The vertical limits (ylim)
    are computed *only* from the data of the requested years.

    Args:
      year_x_day_data: np.ndarray with shape (n_years, 366).
      yr_labels: list of str-years corresponding to the rows.
      years: an int, a range, or list of years to plot.
      interpolate_gaps (bool, optional): Fill in missing values gaps on the plots. Defaults to True.
      pad: fractional padding to add above/below data.

    Returns:
      fig, axes
    """
    if not interpolate_gaps:
        year_x_day_data = np.where(
            year_x_day_data == 0, np.nan, year_x_day_data)
    if isinstance(years, int):
        year_list = [years]
    else:
        year_list = list(years)

    year_strs = [str(y) for y in year_list]
    try:
        idx = [yr_labels.index(ys) for ys in year_strs]
    except ValueError:
        missing = set(year_list) - set(map(int, yr_labels))
        raise ValueError(f"Years not found in labels: {sorted(missing)}")

    # Compute y-limits from *only* those rows
    subset = year_x_day_data[idx, :]
    # Compute raw min/max on the subset
    y_min = np.nanmin(subset)
    y_max = np.nanmax(subset) 
    y_min = min(y_min, 0)   # Force zero into the bracket
    y_max = max(y_max, 0)   
    span = y_max - y_min    # Pad if applicable
    y_limits = (y_min - pad*span, y_max + pad*span)
    n = len(year_list)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(FIG_SIZES["lineplots"][0],
                 FIG_SIZES["lineplots"][1]*n),
        sharex=True
    )
    if n == 1:
        axes = [axes]
    # plot each year
    for ax, row, yr in zip(axes, idx, year_list):
        ax.plot(DAY_RANGE, year_x_day_data[row], color="white", lw=1)
        ax.axhline(0, color="skyblue", ls="--", lw=0.8)
        ax.set_ylim(*y_limits)
        ax.set_facecolor("black")
        ax.set_title(f"Year {yr}", color="white")
        ax.tick_params(colors="white")
        ax.set_xticks(MONTH_TICKS["positions"])
        ax.set_xticklabels(MONTH_TICKS["labels"], color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
    # shared X-label and layout
    fig.text(0.5, 0.04, "Month", ha="center", color="white")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def interpolate_yearly(year_x_day_data: np.ndarray) -> np.ndarray:
    """
    Subfuntion | Given an array (n_years x n_days) with NaNs marking 
    missing days,return a new array where each row has been linearly
    interpolated along the day-axis.
    """
    # Build a DataFrame so we can use DataFrame.interpolate()
    df = pd.DataFrame(
        year_x_day_data,
        index=[f"yr_{i}" for i in range(year_x_day_data.shape[0])],
        columns=range(year_x_day_data.shape[1])
    )

    # interpolate across columns (i.e. days), limit_direction='both'
    df_interp = df.interpolate(
        method="linear",
        axis=1,
        limit_direction="both"
    )

    return df_interp.to_numpy()


def plot_stable_years(
    year_x_day_data: np.ndarray,
    labels: List[str],
    n_years: int = 3,
    interpolate_gaps: bool = False,
    figsize_per_row: float = 2.0
) -> None:
    """
    Select and plot the N most stable years as a joyplot.

    Given a 2D array (years x days) and a list of year-labels, pick the N
    rows with lowest stddev (ignoring zero sentinels), then optionally
    interpolate those rows, and finally display a joyplot.

    Args:
        year_x_day_data (np.ndarray):
            A 2D array of shape `(n_years, n_days)`, where zeros mark missing
            data points. Each row corresponds to one year, each column to one day.
        labels (List[str]):
            A list of length `n_years` of human-readable labels (e.g. year strings)
            for each row in `year_x_day_data`.
        n_years (int, optional):
            Number of years (rows) to select based on lowest standard deviation.
            Defaults to 3.
        interpolate_gaps (bool, optional):
            If True, perform linear interpolation on NaN gaps *only* in the
            selected subset of years before plotting. Defaults to False.
        figsize_per_row (float, optional):
            Vertical size in inches allocated to each ridgeline. The final
            figure height will be `n_years * figsize_per_row`. Defaults to 2.0.

    Returns:
        None: Displays the plot inline (e.g. in a Jupyter notebook) but does not
        return any object.

    Raises:
        ValueError:
            If `len(labels)` does not match the number of rows in
            `year_x_day_data`.
    """
    # Mask zeros → NaN, for both std‐calculation AND later plotting
    masked_all = np.where(year_x_day_data == 0, np.nan, year_x_day_data)
    # Compute stddev on the raw masked data (no interpolation here!)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stds = np.nanstd(masked_all, axis=1, ddof=0)
    # Pick the n_years smallest‐std rows
    chosen_idx = np.argsort(stds)[:n_years]
    chosen_idx_sorted = sorted(chosen_idx, key=lambda i: stds[i], reverse=True)
    # Extract just those rows
    data_sel_raw = masked_all[chosen_idx_sorted, :]
    labels_sel = [labels[i] for i in chosen_idx_sorted]
    if interpolate_gaps:
        data_sel = interpolate_yearly(data_sel_raw)
    else:
        data_sel = data_sel_raw

    fig_height = n_years * figsize_per_row
    fig, axes = joyplot(
        data_sel.tolist(),
        labels=labels_sel,
        x_range=DAY_RANGE,
        kind="values",
        linewidth=1.0,
        overlap=0.8,
        fill=False,
        figsize=(10, fig_height),
        background="black"
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98,
                        bottom=0.05, hspace=0.3)

    # Annotate each ridgeline
    days = np.array(list(DAY_RANGE))
    for arr, ax in zip(data_sel, axes):
        arr = np.array(arr, dtype=float)
        if np.all(np.isnan(arr)):
            continue

        # highlight min/max and median
        min_i, max_i = np.nanargmin(arr), np.nanargmax(arr)
        ax.scatter(days[min_i], arr[min_i], color="orange", s=30)
        ax.scatter(days[max_i], arr[max_i], color="cyan",   s=30)
        ax.text(days[min_i], arr[min_i], f"{int(arr[min_i])} mm",
                color="orange", ha="right", va="top",    fontsize=8)
        ax.text(days[max_i], arr[max_i], f"{int(arr[max_i])} mm",
                color="cyan",   ha="left",  va="bottom", fontsize=8)
        med = np.nanmedian(arr)
        ax.hlines(med, days[0], days[-1],
                  colors="yellow", linestyles=":", linewidth=0.8, alpha=0.7)
        ax.text(days[-1] + 2, med, f"med={med:.1f} mm",
                color="yellow", va="center", fontsize=7)
        ax.set_xlim(days[0], days[-1])
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Hauteur")
        ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.show()


def plot_diverging_years(
    year_x_day_data: np.ndarray,
    yr_labels: List[str],
    years: List[int],
    interpolate_gaps: bool = True,
    y_limits: tuple[int, int] = (-1000, 9000)
) -> None:
    """
    Plot line chart of daily-max values for specified years.

    Args:
        year_x_day_data (np.ndarray): shape (n_years, 366), with NaNs for missing days
        yr_labels (List[str]): list of years (as strings) mapping to rows of data
        years (List[int]): which years to plot
        interpolate_gaps (bool): if True, linearly interpolate NaN gaps
        y_limits (tuple): (ymin, ymax) for all subplots
    """
    # map year → row index
    year_to_idx = {int(y): i for i, y in enumerate(yr_labels)}
    idxs = [year_to_idx[y] for y in years]
    sel = year_x_day_data[idxs, :].astype(float)

    if interpolate_gaps:    # Fill all NaNs
        sel = interpolate_yearly(sel)
    else:
        # break the line on zeros
        sel = np.where(sel == 0, np.nan, sel)

    fig, axes = plt.subplots(
        nrows=len(years),
        ncols=1,
        figsize=(
            FIG_SIZES["lineplots"][0],
            FIG_SIZES["lineplots"][1] * len(years),
        ),
        sharex=True
    )
    if len(years) == 1:
        axes = [axes]
    for ax, yr, row in zip(axes, years, sel):
        ax.plot(DAY_RANGE, row, color="white", lw=1)
        ax.axhline(0, color="skyblue", ls="--", lw=0.8)
        ax.set_ylim(*y_limits)
        ax.set_facecolor("black")
        ax.set_title(f"Year {yr}", color="white")
        ax.tick_params(colors="white")
        ax.set_xticks(MONTH_TICKS["positions"])
        ax.set_xticklabels(MONTH_TICKS["labels"], color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.5, 0.04, "Month", ha="center", color="white")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def animate_years(year_x_day_data: np.ndarray,
                  yr_labels: List[str],
                  interval_ms: int = 1000, output: str = "cinema.gif",
                  interpolate_gaps: bool = True,
                  truncated: bool = False) -> None:
    """
    This function generates an animated GIF that sequentially displays daily maximum 
    water levels for each year in the dataset. The animation provides a dynamic 
    representation of how water levels change throughout different years.

    Args:
        year_x_day_data (np.ndarray): A 2D array of shape `(n_years, n_days)`. Each row 
        corresponds to one year, each column to one day.       
        labels (List[str]): A list of length `n_years` of human-readable labels
        for each row in `year_x_day_data`.        
        interval_ms (int, optional): Milliseconds between animation frames. 
        Controls the speed of the animation. Defaults to 1000 (1 second).       
        output (str, optional): Filename for the output animated GIF. Defaults to "cinema.gif".
        truncated (bool, optional): If True, limits the y-axis to the 99th percentile 
        of water levels to focus on most common ranges. 
        If False, uses the full range of water levels. Defaults to False.

    Returns:
        None. Saves an animated GIF.

    Notes:
        - The function uses matplotlib's animation capabilities.
        - Each frame represents a year's daily maximum water levels.
        - The animation cycles through all years in the dataset.

    Usage:
        >>> animate_years(dataframe, interval_ms=500, output='filename.gif')
    """
    # Reverse so we go oldest → newest
    year_x_day_data = year_x_day_data[::-1]
    yr_labels = yr_labels[::-1]
    # cast and break lines on zeros
    sel = year_x_day_data.astype(float)
    sel = np.where(sel == 0, np.nan, sel)
    if interpolate_gaps:
        sel = interpolate_yearly(sel)
    n_years, n_days = sel.shape
    days = np.arange(1, n_days + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot([], [], lw=2)
    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")
    ax.set_xlim(1, n_days)
    flat = sel.flatten()
    finite = flat[np.isfinite(flat)]
    if truncated:
        ymin, ymax = np.nanpercentile(finite, 0), np.nanpercentile(finite, 99)
    else:
        ymin, ymax = finite.min(), finite.max()
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Max water level")

    # animation callbacks
    def init():
        line.set_data([], [])
        title.set_text("")
        return line, title
    
    def animate(frame_idx: int):
        y = sel[frame_idx, :]
        line.set_data(days, y)
        title.set_text(f"Year: {yr_labels[frame_idx]}")
        return line, title

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_years,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )
    ani.save(output, writer=animation.PillowWriter(fps=1000 // interval_ms))
    plt.close(fig)
    print(f"Saved animation to {output}")
