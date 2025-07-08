# DataSwag, the data exploration and processing notebook

**Oh liquid chronology! Oh rainfall's whispers and drought's silent scream!**

![Banner Image](<img/déluge.jpeg> "A painting depicting a shipwreck, afterwards. A storm is looming in the background.")
<br>*Detail of "scène du déluge" by Théodore Géricault ([Louvre collection](https://collections.louvre.fr/ark:/53355/cl010064842)) | ♫ Hobson's Choice - Heligoland*

## Genesis

> I was wandering on the shores of the internet, gasp-hoping for a fresh breath of inspiration and stepped upon that [map](retrouvelelien) washed ashore the forgotten bytescape. I didn't knew about joyplots before -and instantly became curious about it. A digicrush? I just fall upon its minimalistic, very 80s, swag and decided to work out something around it. That became the *DataSwag* making-of notebook about my submission to the HackaViz 2025 competition.

## Features

This repository provides a collection of Python utilities to download, process, analyze and visualize long term water-level data. It features:

*Part I: exploration & processing*

- Data download and loading from Parquet files
- Exploratory plots: histograms, density plots, boxplots..
- Missing-data detection and reporting
- Outlier detection via Z-score analysis
- Data preparation: aggregation and pivoting to a “years x days” array

*Part II: dataviz*

- joyplots (ridge plots) with custom styles
- Line-plots for selected years
- Annotated ridgelines providing specific insights about the dataset
- Animated GIF generation of year-by-year time series

## Table of contents

<details>
<summary>Contents - click to expand</summary>

- [DataSwag, the data exploration and processing notebook](#dataswag-the-data-exploration-and-processing-notebook)
  - [Genesis](#genesis)
  - [Features](#features)
  - [Table of contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [The dataset](#the-dataset)
    - [hauteur\_eau\_9\_crues.parquet](#hauteur_eau_9_cruesparquet)
      - [Description](#description)
      - [Sample data](#sample-data)
    - [hauteur\_eau\_serie\_longue\_toulouse.parquet](#hauteur_eau_serie_longue_toulouseparquet)
      - [Description](#description-1)
      - [Sample data](#sample-data-1)
  - [Configuration](#configuration)
  - [Utility functions](#utility-functions)
    - [File loading](#file-loading)
    - [Data quality checks](#data-quality-checks)
      - [Missing data and periods](#missing-data-and-periods)
      - [Outlier Detection](#outlier-detection)
      - [Distributions and frequencies](#distributions-and-frequencies)
    - [Data preparation](#data-preparation)
  - [The domain of the plots](#the-domain-of-the-plots)
    - [Classic remix: recreating a legend](#classic-remix-recreating-a-legend)
    - [The apex: tailor-made, custom flavored, ridgelines](#the-apex-tailor-made-custom-flavored-ridgelines)
    - [Temporal selections: yearly line-plots](#temporal-selections-yearly-line-plots)
    - [Converging or adrift spectrums: grasping some insights](#converging-or-adrift-spectrums-grasping-some-insights)
    - [Animation time](#animation-time)
  - [Limitations and potential improvements](#limitations-and-potential-improvements)
  - [Going beyond](#going-beyond)
  - [Contributing](#contributing)
  - [License](#license)
    - [Disclaimer](#disclaimer)
  - [Acknowledgments](#acknowledgments)

</details>

## Requirements

- Python (3.10+). Installation [tips](https://askubuntu.com/questions/682869/how-do-i-install-a-different-python-version-using-apt-get)
- Jupyter notebooks. Installation [tips](https://jupyter.org/install)

*Dependencies:*

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- pyarrow
- requests
- joypy

## Installation

Please make sure you're using at least Python 3.10. The code might run with older python versions (down to 3.8) but all the features have not been tested so far.

*(Recommended)* Only if you're using Python versions prior to 3.10 on your system. On Debian/Ubuntu:

```bash
sudo apt install python3.10
```

*(Optional)* On Unix systems, create and activate a virtual environment:

```bash
python3.10 -m venv dataswag # can be 3, 3.10.. replace with your version.
source dataswag/bin/activate
```

Clone the repository and navigate to the processing folder:

```bash
git clone https://github.com/brooks-code/miniature-fortnight.git
cd miniature-fortnight/processing
```

Install dependencies (see [Requirements](#requirements)):

```bash
pip install -r requirements.txt
```

Enjoy the jupyter notebook (`dataswag_dataviz.ipynb`).

## The dataset

Datasets are served via GitHub (an archive is available in the archive folder).

```markdown
hauteur_eau_9_crues.parquet: Data about the nine major flood events.
hauteur_eau_serie_longue_toulouse.parquet: 167-year continous water-levels daily series for the Toulouse area.
```

### hauteur_eau_9_crues.parquet

This dataset contains information about water-levels and flood events recorded at specific stations (identified by their code). The columns in the dataset are as follows:

#### Description

| Column Name  | Data Type | Description                                      |
| ------------ | --------- | ------------------------------------------------ |
| code_station | String    | Identifier for the monitoring station.           |
| hauteur      | String    | Water-level measured in millimeters (mm).        |
| date_heure   | DateTime  | Timestamp of the measurement in ISO 8601 format. |
| code_crue    | Integer   | Year of the recorded flood event.                |

Where `code_station` is the measuring station code, `hauteur` the water-level, `date_heure` the date and time, and `code_crue` the flood code.

#### Sample data

| code_station | hauteur   | date_hauteur         | code_crue |
| ------------ | --------- | -------------------- | --------- |
| O200004002   | 820 [mm]  | 1879-04-01T12:00:00Z | 1879      |
| O200004002   | 1300 [mm] | 1879-04-02T06:00:00Z | 1879      |

### hauteur_eau_serie_longue_toulouse.parquet

This dataset contains long-term water-level measurements (over a 167 years span) recorded at specific measuring stations. The columns in the dataset are as follows:

#### Description

| Column Name  | Data Type | Description                                      |
| ------------ | --------- | ------------------------------------------------ |
| code_station | String    | Identifier for the monitoring station.           |
| hauteur      | Float     | Water-level measured in meters.                  |
| date_hauteur | DateTime  | Timestamp of the measurement in ISO 8601 format. |

Where `code_station` is the measuring station code, `hauteur` the water-level and `date_heure` the date and time,

#### Sample data

| code_station | hauteur | date_heure                |
| ------------ | ------- | ------------------------- |
| O200004001   | 600.0   | 1946-01-01 11:00:00+00:00 |
| O200004001   | 600.0   | 1946-01-02 11:00:00+00:00 |

## Configuration

Most parameters can be adjusted by passing optional arguments:

- Customize constants at top of module (e.g.: `Z_THRESH`, `FIG_SIZES`...)
- Dataset folder & filename filter: `load_flood_dataset(folder, delimiter)`
- Z-score threshold: `detect_outliers(z_thresh=…)`
- Aggregation columns: e.g.: `prepare_data(dt_col=…, water_lev_col=…)`
- Styles: four pre-defined keys are available in `plot_byflavor()`: classic, calm, excess and occitania
- Animation: frame interval, output filename, y-axis truncation in `animate_years()`

## Utility functions

### File loading

Looks for any Parquet files whose filenames contain a delimiter keyword (`DELIMITER`) in a specified directory (creates it if it doesn't exists), and—if none are found—iterates over a dict of known URLs (`DATASET_URLS`), downloading each. It then re-scans the folder, picks the first matching Parquet file, reads it into a pandas DataFrame via PyArrow, and returns both the DataFrame and the file path.

*usage:*

```python
from hackaviz_helper import load_flood_dataset

dataset_folder = "folder"
DELIMITER = "keyword"

df, filepath = load_flood_dataset(dataset_folder, DELIMITER)
```

### Data quality checks

#### Missing data and periods

`find_missing_periods(df, dt_col)` scans a time‐series column in a DataFrame to determine which years, months, and days are absent between the earliest and latest timestamps, returning a dict of missing periods. `plot_missing_values(df, dataset_path)` prints overall null‐count statistics, lists any rows with missing data, and—for each column that has nan values, renders a heatmap of the missing‐value pattern; it also computes pairwise null‐value correlations to suggest whether the missingness is likely at random (MAR) or not at random (MNAR).

*usage:*

```python
from hackaviz_helper import find_missing_periods, plot_missing_values

find_missing_periods(df, "datetime_col")
plot_missing_values(df, dataset_filepath)
```

#### Outlier Detection

`detect_outliers(df, dt_col, water_lev_col, z_thresh, min_std)` flags anomalous readings in a time‐series by first extracting the date from each timestamp and then grouping measurements by station code and date. It computes each group’s count, mean, and standard deviation; merges these stats back into the original data; and calculates a Z-score for every observation. Any record belonging to a sufficiently large and variable group (n > 1, std ≥ min_std) whose |Z-score| exceeds `z_thresh` is labeled an outlier. The function also prints how many days contained outliers and returns a DataFrame of all detected outlier rows.

*usage:*

```python
from hackaviz_helper import detect_outliers

detect_outliers(df, "datetime_col", "water_lev_col", Z_THRESH, MIN_STD)
```

#### Distributions and frequencies

`plot_frequency_histogram(df)` takes every numeric column in a DataFrame and draws a side-by-side frequency histogram for each.
`plot_distribution(df, grid)` goes deeper: for each numeric column it produces four plots—histogram, KDE, Q-Q plot, and boxplot—either in a 2x2 grid or as separate figures, then prints key summary statistics (mean, median, mode, standard deviation, skewness, kurtosis) below.

*usage:*

```python
from hackaviz_helper import plot_frequency_histogram, plot_distribution

plot_frequency_histogram(df)
plot_distribution(df, grid=True)
```

### Data preparation

`prepare_data(df, dt_col, water_lev_col)` first normalizes timestamps to midnight, then builds a complete daily index spanning from January 1 of the earliest year to December 31 of the latest. It groups the original readings by day to get each day’s maximum water level, merges that into the full index (inserting NaNs where days had no data), flags days containing no records with zero sentinels, and finally pivots into a 2D array of shape (years x days). It returns this array, a list of year labels, and the merged daily DataFrame.
`interpolate_yearly(data)` is a subfunction that takes that (years x days) array with NaNs and, via a pandas DataFrame, linearly interpolates missing values along each row (day-axis), returning a fully filled NumPy array.

> [!NOTE]
> The data prep has been crafted with a DRY philosophy in mind. **Its outputs are used by most of the plotting functions below**.

*usage:*

```python
from hackaviz_helper import prepare_data

data_by_year, yr_labels, daily_df = prepare_data(df)

# data_by_year: NumPy array of shape (n_years, 366) (zero-padded missing days)
# yr_labels: List of years as strings (descending: newest first)
# daily_df: DataFrame with columns: date, hauteur, is_zero, Year, Day
```

## The domain of the plots

```markdown
    Classic joyplot – plot_original_joyplot()
    Styled joyplot – plot_byflavor()
    Line-lots (selected years) – plot_years()
    Stable years joyplot – plot_stable_years()
    Diverging multi-year – plot_diverging_years()
    Animated GIF – animate_years()
```

### Classic remix: recreating a legend

`plot_original_joyplot(df, dt_col, water_lev_col)` generates a classic “joyplot” of binned (by counts) daily maximum water levels over time, with each year rendered as an overlapping density‐style ridge. It first normalizes the timestamp column to dates and computes the per-day maximum of the specified water‐level column, tagging each record with its year. Years are ordered from newest (top) to oldest (bottom), and the global min/max water levels set the x-axis range. Finally, it calls `joyplot()` with a customized styling (black background, white lines, no fills or grid, and bins data counts) to display the seasonal distribution of daily peaks year by year.

*usage:*

```python
from hackaviz_helper import plot_original_joyplot

plot_original_joyplot(df)
```

### The apex: tailor-made, custom flavored, ridgelines

`plot_byflavor(data, yr_labels, daily_metrics_df, style_key)` renders a fully styled ridge-plot of a daily time-series array, coloring and highlighting each year according to a chosen “flavor” from the `STYLES` config. It first computes a flood metric per year (e.g.: annual max or percentile), normalizes it (if required), and builds a color map so that years with higher values get more intense hues. Then it calls `joyplot()` on the 2D data (years x days) without fill, applies per-row line-color and line-width settings (thicker for highlighted years), overplots any zero-value points in the background color to de-emphasize missing days, and finally adjusts axes limits, face colors, tick colors, and adds a styled title. The result is a visually coherent joyplot that immediately draws the eye to the flood-year signatures. Oldest years are at the bottom and newest ones on top.

> [!IMPORTANT]
> This one was **tricky**. I ran into some issues with the missing periods (days without water level records). Usually, either you interpolate the missing values (connect the dots) or leave line breaks. But here, somehow, joypy couldn't resolve either of these options and it just stacks together the non missing datapoints resulting in a shift and inaccurate lines*. The fix so far is to fill the missing periods with sentinel values (zeroes), so the shift doesn't happen and then replot them in the background color (making them look like invisible). Also we then have to flag the sentinels, in order to avoid mixing them with computations involving the real data. This is *overly complicated* and another cleaner approach should definately be considered.
>
> **Imagine we have a year with with 30 missing days all over the year (2/3 missing records per month). Well, joypy will stack all the values and render a plot that makes it look like 30 days of december are missing.*

*usage:*

```python
from hackaviz_helper import plot_byflavor

plot_byflavor(data_by_year.copy(), labels, daily, style_key='classic')
# data_by_year, labels and daily are computed by prepare_data()
```

### Temporal selections: yearly line-plots

`plot_years(data_by_year, yr_labels, years, interpolate_gaps, pad)` draws one dark‐themed line plot per requested year, showing daily‐max values against day‐of‐year. You can pass a single year, a list/range, and choose whether to interpolate missing‐day gaps (or leave them blank). It first extracts the rows matching years from the (n_years x 366) array, computes Y‐axis limits just from those seasons (always including zero and adding a fractional pad), then creates vertically stacked subplots sharing the X‐axis. Each axis plots the white time series, a light dashed zero line, black background, month tick labels, and a title. The result is a set of comparable, focused seasonal line charts.

*usage:*

```python
from hackaviz_helper import plot_years

plot_years(data_by_year.copy(), labels, 1875, interpolate_gaps=False)
# data_by_year, labels are computed by prepare_data()
```

### Converging or adrift spectrums: grasping some insights

Both plotting utilities take the pre-computed “years x days” array plus its year labels and produce dark‐themed visual comparisons, but they spotlight different seasons. `plot_stable_years` masks out zero sentinels, optionally linearly fills gaps, then computes each year’s standard deviation to pick the N most “stable” seasons. It joyplots those ridgelines, annotating each with its minimum and maximum day values (and a median line), so you immediately see which years had the smallest variability. `plot_diverging_years` instead focuses on a *user-specified* list of years: it maps them to array rows, interpolates or leaves gaps, and draws stacked white line charts of daily peaks against day-of-year (complete with a zero baseline, month ticks, and shared axes) so you can directly compare how flood‐year hydrographs diverge.

*usage:*

```python
from hackaviz_helper import plot_stable_years, plot_diverging_years

plot_stable_years(data_by_year, labels, interpolate=True, n_years=3)
plot_diverging_years(data_by_year, labels, years=[2024, 2020], interpolate_gaps=False)
# data_by_year, labels are computed by prepare_data()
```

### Animation time

`animate_years(data_by_year, yr_labels, interval_ms, output, interpolate_gaps, truncated)` turns a years x days array into a looping GIF that steps through each year’s daily‐max hydrograph in sequence. It reverses the data so the animation runs oldest→newest, masks zero‐sentinel days as gaps, optionally linearly interpolates missing segments, and auto‐scales the y‐axis either to the full data range or up to the 99th percentile. A single matplotlib figure is updated frame‐by‐frame—plotting day‐of‐year vs. water-level and annotating the current year—then exported as a Pillow‐backed GIF at a chosen frame interval.

*usage:*

```python
from hackaviz_helper import animate_years

animate_years(data_by_year, labels, interval_ms=700, output="yearly_animation.gif", interpolate_gaps=False, truncated=True)
# data_by_year, labels are computed by prepare_data()
```

## Limitations and potential improvements

- [joypy issue](#the-apex-tailor-made-custom-flavored-ridgelines): missing days are filled with zero-values to preserve the array shape; the trick so far is that zeroes are masked or interpolated in the plots. *It would be nice to avoid that*.
- Implement more robust missing-data strategies (e.g.: seasonal interpolation, imputation).
- Integrate advanced outlier detection methods (e.g.: Hampel filter, LOF).
- Leap-day handling: all years are represented with 366 columns (Feb 29).
- Performance: large datasets (decades x daily records) can consume significant memory when pivoted.
- Static thresholds: Fixed Z-score and min-std thresholds may miss contextual outliers; it could be better to consider dynamic methods.
- Aggregation, are mostly based on daily maximas. For some computations considering also the median could be more robust.

## Going beyond

I decided to focus on the joyplots early in the challenge but when I was brainstorming my whereabouts at the beginning of the competition, I found these resources to run some predictions:

- [Running a Flood Simulation](https://pypims.readthedocs.io/en/latest/Tutorials/flood.html):
This Jupyter notebook tutorial provides a step-by-step guide for setting up a flood model using the pypims package. It includes generating inputs, running simulations, and plotting results. It covers loading DEM and rainfall data, initializing the flood model, and defining boundary conditions.pypims package.

- [Predicting Flooding with Python](https://www.tobiolabode.com/blog/2020/9/23/predicting-flooding-with-python):
This project focuses on predicting long-term flooding risks by analyzing rainfall data and land elevation. It involves data cleaning, time-series analysis, and forecasting using the statsmodels package. The tutorial walks through data wrangling with pandas, monthly rainfall aggregation, and time-series decomposition.
  
## Contributing

Contributions are welcome! Feel free to open issues or discuss any improvements.
Each contribution and feedback helps improve this project - it's always an honour! ♡

- Fork the repository.
- Create a branch for your feature or bug fix.
- Submit a pull request with your changes.

## License

Project released under the [MIT License](https://mit-license.org/). See [LICENSE](../LICENSE) for details.

### Disclaimer

This project is not affiliated with or endorsed by any third party. It is provided as is for educational and practical uses. Use it at your own risk.

## Acknowledgments

- The [Toulouse dataviz](https://toulouse-dataviz.fr/) team.
- [Leonardo Taccari](https://github.com/leotac), the [joypy package](https://pypi.org/project/joypy/) maintainer.
- This software uses various [open source libraries and Unix utilities](#requirements). Special **thanks** to the developers and the community for providing these valuable tools.

**Datasets:**

- The competition datasets are available in this [repo](https://github.com/Toulouse-Dataviz/hackaviz-2025/tree/main/data). If you plan on using them outside the scope of this project, please contact the repository moderators.
