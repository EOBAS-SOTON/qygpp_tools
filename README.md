# Tools for Dealing with the QY-GPP Dataset

Currently this project contains tools for analysis and display of the QY-GPP dataset, including:

- extracting time series.
- plotting global and local maps.
- generating temporal means and summation maps.
- statistics per landcover class.
- longitudinal graphs of the GPP observed.
- output of basic statistics for a given extend.

## TODO

The following features are planned for when the dataset has been published:

- add method(s) for downloading selected temporal and spatial extends.


## Installation

Can be install by running `pip install .` from the main dir (the one containing
pyproject.toml).
For development, `pip install -e .` is recommended.

It has _not_ been uploaded to the Python Package Index so far and currently
there are no plans to do so.


## Usage

Below some basic usage is presented.
The example is very much not exhaustive.
The full list of individual available functions and what their calling parameters
mean can be found in the script files themselves.
Not everything is documented to a professional level, but care has been taken to
have readable code with sensible variable and function names.

The package is loaded as follows:

```python

from qygpp_tools import analysis

```

All the following examples assume that the data is in a directory structure like this:

```
root/
├── 20200101/
│   ├── QISCARF_GPP_20200101_h00v08_m0.830_b0.053_pc31.000_qi0.080_v02.tif
│   ├── QISCARF_GPP_20200101_h00v09_m0.830_b0.053_pc31.000_qi0.080_v02.tif
│   └── ...
├── 20200109/
│   ├── QISCARF_GPP_20200109_h00v08_m0.830_b0.053_pc31.000_qi0.080_v02.tif
│   ├── QISCARF_GPP_20200109_h00v09_m0.830_b0.053_pc31.000_qi0.080_v02.tif
│   └── ...
└── YYYYMMDD/
    └── files for each tile
```

### Global Composite

To plot a global composite, the function `plot_composite` is used.
It loads data for specified date(s) and generates a map visualization with optional analytics.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | str | *required* | Path to the root directory containing the date-organized data |
| `dates` | list | *required* | List of dates to plot (format: `"YYYY-MM-DD"`). If empty, all available dates are processed |
| `fnpart_out` | str | *required* | Output filename prefix (date will be appended) |
| `title` | str | *required* | Title displayed on the plot |
| `coarsen` | int | `1` | Coarsening factor to reduce resolution (e.g., `8` = 8× downscaling). Useful for RAM-limited machines |
| `vmax` | float | `NaN` | Maximum value for the colour scale. If not set, auto-calculated from data |
| `value_factor` | float | `1.0` | Multiplier applied to all values (useful for unit conversion or to generate comparisons) |
| `max_factor` | float | `1.0` | Factor applied to the auto-calculated maximum (e.g., `0.75` caps at 75% of max) |
| `extra_actions` | str | `''` | Additional outputs, combine with `+`: `"plot"`, `"record"`, `"sum"`, `"mean"` (see below) |
| `fn_worldcover` | str | `''` | Path to WorldCover dataset for land cover percentage analysis. Given LC is reprojected & resolution matched to the plot data |
| `lon_min/lon_max` | float | `NaN` | Longitude bounds for cropping the plot |
| `lat_min/lat_max` | float | `NaN` | Latitude bounds for cropping the plot |
| `extra_xlabel` | str | `''` | Additional unit label (e.g., `"year"` → `gC/m²/year`) |
| `exclude_lc_class` | list | `[]` | Land cover classes to exclude from analysis (e.g., `["Snow and Ice"]`) |
| `colourscale` | str | `'viridis'` | Matplotlib colourscale name |
| `export_stats` | bool | `False` | Export min/max/mean/sum statistics to a text file |
| `save_composite` | bool | `False` | Save the processed dataset to disk |
| `stop_after_save` | bool | `False` | Stop processing after saving (skip plotting) |

#### Extra Actions

The `extra_actions` parameter controls additional outputs beyond the main map plot. Multiple actions can be combined using `+` (e.g., `"record+plot+sum+mean"`).

| Action | Description |
|--------|-------------|
| `plot` | Generate longitude profile plots (GPP values aggregated by latitude) |
| `record` | Save the longitude profile data to CSV files for further analysis |
| `sum` | Create a longitude profile using the **sum** of values at each latitude |
| `mean` | Create a longitude profile using the **mean** of values at each latitude |

**How it works:**
- When `plot` or `record` is present, longitude profile plots are generated
- `sum` and `mean` determine *how* values are aggregated across longitudes
- If `fn_worldcover` is provided, a land cover percentage pie chart is also generated
- Data files are saved to a `data_save/` subdirectory next to the output plot

**Example combinations:**
- `"plot+sum"` — Generate and display only the sum-based longitude profile
- `"record+plot+sum+mean"` — Generate both sum and mean profiles, display them, and save the data
- `"record+mean"` — Save mean profile data without displaying the plots

#### Example

```python
# Plot a single dataset composite
analysis.plot_composite(
    directory="/dir/to/dataset/",
    dates=["2020-01-01"],
    fnpart_out="output/composite_",
    title="QY-GPP GPP Composite - Test",
    coarsen=8,                                 # 8× downscaling for RAM-limited machines 
    vmax=40.0,                                 # maximum of legend colour scale
    extra_actions="record+plot+sum+mean",      # generate longitude plots and save data
    fn_worldcover="WorldCover_500m_2021.tif",  # land cover percentage analysis
    colourscale="viridis",                     # colour scheme
    export_stats=True                          # export statistics to file
)
```


### Global Map, Combined Composites

To plot aggregated data over a date range (e.g., the sum or mean of yearly GPP), the `plot_range` function is used.
It combines multiple composites into a single visualization with temporal aggregation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | str | *required* | Path to the root directory containing the date-organized data |
| `dates` | list | *required* | Date range as `["YYYY-MM-DD", "YYYY-MM-DD"]` (start, end) |
| `file_out` | str | *required* | Output filename for the plot (must end with `.png`) |
| `title` | str | *required* | Title displayed on the plot |
| `coarsen` | int | `1` | Coarsening factor to reduce resolution (e.g., `8` = 8× downscaling) |
| `operator` | str | `"mean"` | Temporal aggregation: `"mean"`, `"sum"`, or `"only_plot"` |
| `low_mem` | bool | `False` | Use disk-based processing for large date ranges (recommended for yearly/seasonal) |
| `tmp_dir` | str | `"tmp/"` | Temporary directory for `low_mem` mode intermediate files |
| `vmax` | float | `NaN` | Maximum value for the colour scale |
| `output_data` | bool | `False` | Save the aggregated dataset to disk (as `.zarr`) |
| `value_factor` | float | `1.0` | Multiplier applied to all values (e.g., `8.0` for MODIS comparison) |
| `continue_plot` | bool | `False` | Skip data loading and use previously saved intermediate files |
| `extra_actions` | str | `''` | Additional outputs: `"plot"`, `"record"`, `"sum"`, `"mean"` (see Global Composite section) |
| `fn_worldcover` | str | `''` | Path to WorldCover dataset for land cover analysis |
| `lon_min/lon_max` | float | `NaN` | Longitude bounds for cropping |
| `lat_min/lat_max` | float | `NaN` | Latitude bounds for cropping |
| `extra_xlabel` | str | `''` | Additional unit label (e.g., `"year"` → `gC/m²/year`) |
| `max_factor` | float | `NaN` | Factor applied to the auto-calculated maximum |
| `exclude_lc_class` | list | `[]` | Land cover classes to exclude from analysis |
| `colourscale` | str | `'viridis'` | Matplotlib colourscale name |
| `export_stats` | bool | `False` | Export statistics to a text file |
| `stop_after_save` | bool | `False` | Stop after saving dataset (skip plotting) |

#### Operator vs Extra Actions

- **`operator`** controls *temporal* aggregation (how composites are combined over time)
- **`extra_actions`** controls *spatial* aggregation for longitude profiles (see Global Composite section)

#### Example

```python
# Plot the accumulated GPP for 2020
analysis.plot_range(
    directory="/dir/to/dataset/",
    dates=["2020-01-01", "2020-12-31"],
    file_out="output/data_range_2020_accumulated.png",
    title="Plot sample for 2020 (accumulated GPP)",
    coarsen=8,
    operator="sum",                             # sum all composites in the date range
    low_mem=True,                               # use disk for intermediate results (recommended)
    value_factor=8.0,                           # scale factor for MODIS comparison
    extra_actions="record+plot+mean",           # generate longitude profile plots
    output_data=True,                           # save the aggregated dataset
    fn_worldcover="WorldCover_500m_2021.tif",   # land cover percentage analysis
    extra_xlabel="year",                        # label as gC/m²/year
    max_factor=0.75                             # cap colour scale at 75% of max
)
```

### Combination: Plot Mean across Years

Currently, no function is implemented to do two processing steps at once.
For example, plotting the mean across several years needs to be done manually with code like below:

```python

year_start = 2020
year_end = 2023

# Step 1: Generate and save yearly sums
for year in range(year_start, year_end+1):
    analysis.plot_range(
        directory="/dir/to/dataset/",
        dates=[f"{year}-01-01", f"{year}-12-31"],
        file_out=f"output/data_range_{year}_accumulated.png",
        title="",
        operator="sum",
        low_mem=True,
        value_factor=8.0,
        output_data=True,
        stop_after_save=True   # skip plotting, only save the data
    )

# Step 2: Load all yearly datasets
years = range(year_start, year_end+1)
list_ds = []

with tqdm(total=len(years)) as pbar:
    for year in years:
        # Load the saved zarr dataset
        ds = xr.open_mfdataset(
            f"output/data_save/data_range_{year}_accumulated.zarr",
            combine='by_coords',
            chunks={"band": 1, "x": 800, "y": 800},
            parallel=True
        )
        
        # Add year as time coordinate
        ds = ds.assign_coords(time=datetime(year=year, month=1, day=1))
        list_ds.append(ds)
        pbar.update(1)

# Step 3: Combine and plot the multi-year mean
ds_combined = xr.concat(list_ds, dim='time')

analysis.plot_dataset(
    ds_combined,
    title=f"Mean Annual GPP ({year_start}-{year_end})",
    file_out="output/mean_annual_gpp.png",
    operator="mean",                         
    coarsen=8,
    value_factor=1.0,                       
    extra_actions="record+plot+mean",
    fn_worldcover="WorldCover_500m_2021.tif",
    extra_xlabel="year",
    vmax=2500,                                # adjust based on your data
    exclude_lc_class=["Permanent Water Bodies", "Built-up", "Snow and Ice"],
    export_stats=True
)
```

### Time Series Extraction

The `extract_timeseries` function extracts GPP values over time for specific point locations (e.g., flux tower sites).
It reads coordinates from a GeoPackage or Shapefile and outputs CSV files with temporal data for each point.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | str | *required* | Path to the root directory containing the date-organized data |
| `dir_out` | str | *required* | Output directory for the extracted CSV files |
| `fn_points` | str | *required* | Path to GeoPackage/Shapefile containing point locations |
| `point_layer` | str | `''` | Layer name within the GeoPackage (required for multi-layer files) |
| `name_column` | str | `''` | Column name to use for naming output files (e.g., station ID) |
| `time_start` | str | `''` | Start date for extraction (`"YYYY-MM-DD"`). If empty, uses earliest available |
| `time_end` | str | `''` | End date for extraction (`"YYYY-MM-DD"`). If empty, uses latest available |
| `scale_factor` | float | `NaN` | Scale factor for int16 TIF data (usually not needed for float data) |
| `flag_value` | bool | `True` | Extract the GPP data values |
| `flag_quality` | bool | `False` | Extract the quality flag values |
| `extract_area` | bool | `False` | Extract mean over a surrounding area instead of single pixel |
| `cellbuffer_extend` | int | `0` | Number of cells to extend around point when `extract_area=True` |

#### Area Extraction

When `extract_area=True`, instead of extracting a single pixel value, the function extracts a rectangular area around each point and computes the mean. The `cellbuffer_extend` parameter controls the size:

- `cellbuffer_extend=0` — single pixel (1×1)
- `cellbuffer_extend=1` — 3×3 pixel area
- `cellbuffer_extend=2` — 5×5 pixel area

This is useful for comparing satellite data with flux tower footprints, which typically cover more than one pixel.

#### Example

```python
# Extract time series for flux tower locations
analysis.extract_timeseries(
    directory="/dir/to/dataset/",
    dir_out="fluxtower_timeseries/",
    fn_points="fluxtower_metadata.gpkg",
    point_layer="AmeriFlux",                  # layer name in GeoPackage
    name_column="Station",                    # column with station names
    time_start="2020-01-01",
    time_end="2020-12-31",
    flag_value=True,                          # extract GPP values
    flag_quality=False,                       # skip quality flags
    extract_area=True,                        # use area mean instead of single pixel
    cellbuffer_extend=1                       # 3×3 pixel area (±1 cell)
)
```

#### Output

For each point in the input file, a CSV file is created in `dir_out` containing:
- Date/time column
- Extracted GPP value (if `flag_value=True`)
- Quality flag (if `flag_quality=True`)

This is useful if the computer used for extraction runs out of RAM.
The recommendation is to extract year-by-year and then concatenate in e.g. Pandas.


## Limitations and Future Enhancements

As with any software package, certain limitations exist along with opportunities for future improvements.

### Memory Usage

This library can be memory-intensive, both during pre-processing and plotting.
**A minimum of 32 GB RAM is recommended, with 64 GB preferred for comfortable operation.**

#### Pre-processing

For temporal aggregation (sum/mean operations), the library uses [Dask](https://www.dask.org/) for delayed/chunked loading.
Based on experience with a 128 GB RAM machine, it is advisable to enable the `low_mem=True` flag when processing yearly or seasonal composites.
This mode processes each tile individually before assembling the final mosaic, significantly reducing peak memory usage.

#### Plotting

The library automatically selects between two matplotlib plotting routines based on image size:

| Pixel Count | Method | Characteristics |
|-------------|--------|-----------------|
| ≤ 25 million | `pcolormesh` | Higher quality, preserves all pixels without resampling |
| > 25 million | `imshow` | More memory-efficient, but applies internal coarsening |

This threshold exists because even a 128 GB RAM machine cannot render a global composite using `pcolormesh` with a coarsening factor of 40 (i.e., 500 m → 20 km resolution).
The trade-off is that `imshow` can produce blockier images for smaller regional extracts.

### Plotted Projection

The output map projection currently switches between Robinson projection (optimised for global plotting) and Miller projection (better at local plots).
Manually setting the projection is a possible future enhancement.

### Extra Plotting Data

Currently, the plots always contain country borders and coastline.
In future there might be options to add finer subdivisions like local government areas and rivers to plots.
This would add extra locality for especially local plots.
