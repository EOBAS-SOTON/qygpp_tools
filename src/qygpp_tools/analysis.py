#
# Routines for analysis of the QY-GPP dataset.
#
# (C) 2025, Sven Berendsen, s.berendsen@soton.ac.uk
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import datetime
import gc
import shutil

import geopandas as gpd
import numpy as np
import os
from tqdm import tqdm
import xarray as xr
import zarr

from .analysis_core import plotting as _plot
from .analysis_core import timeseries as _ts


def extract_timeseries(directory: str,
                       dir_out: str,
                       fn_points: str,
                       point_layer: str = '',
                       name_column: str = '',
                       time_start: str = '',
                       time_end: str = '',
                       scale_factor: float = np.nan,
                       flag_value: bool = True,
                       flag_quality: bool = False,
                       cellbuffer_extend: int = 0,
                       extract_area: bool = False) -> None:
    """
    Extract a timeseries from a directory of SEN4GPP output.

    :param directory: str, directory containing the output files
    :param fn_out: str, filename for the output table
    :param fn_points: str, filename to take the points from
    :param name_layer: str, name of the layer to take the points from
    :param time_start: str, start time for the extraction
    :param time_end: str, end time for the extraction
    :param scale_factor: float, scale factor for the data for int16 tif only
    :param flag_value: bool, flag to extract the value data (default: true)
    :param flag_quality: bool, flag to extract the quality data (default: false)
    :param cellbuffer_extend: int, mean of the extended cellbuffer (default: 0)
    """

    # get list of subdirectories in directory if it is of the form YYYY-MM-DD
    subdirs = [
        f for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]
    subdirs = [
        x for x in subdirs if len(x) == 10 and x[4] == '-' and x[7] == '-'
        and x[0:4].isdigit() and x[5:7].isdigit() and x[8:10].isdigit()
    ]
    subdirs = sorted(subdirs)

    # get the time range
    if time_start == '':
        t_start = datetime.datetime.strptime(subdirs[0], "%Y-%m-%d")
    else:
        t_start = datetime.datetime.strptime(time_start, "%Y-%m-%d")

    if time_end == '':
        t_end = datetime.datetime.strptime(subdirs[-1], "%Y-%m-%d")
    else:
        t_end = datetime.datetime.strptime(time_end, "%Y-%m-%d")

    # get the subdirectories in the time range
    subdirs = [
        f for f in subdirs
        if datetime.datetime.strptime(f, "%Y-%m-%d") >= t_start
        and datetime.datetime.strptime(f, "%Y-%m-%d") <= t_end
    ]

    # convert x & y into target coordinates
    gdf = gpd.read_file(fn_points, layer=point_layer)

    # safety
    unique_geom_types = gdf.geom_type.unique()
    if len(unique_geom_types) != 1 or unique_geom_types[0] != 'Point':
        raise ValueError("Only Point geometries are supported")

    print(f"\nExtracting {len(gdf)} points from "
          f"{len(subdirs)} timesteps, loading data...")

    # get the filenames
    # TODO support .nc as well
    list_ds = []
    print(f"\nLoading {len(subdirs)} subdirectories / times...")
    with tqdm(total=len(subdirs)) as pbar:
        for subdir in subdirs:
            files = [
                os.path.join(directory, subdir, f)
                for f in os.listdir(os.path.join(directory, subdir))
                if os.path.isfile(os.path.join(directory, subdir, f))
                and f.endswith('.tif')
            ]

            # get the data
            ds = xr.open_mfdataset(files,
                                   combine='by_coords',
                                   chunks={
                                       "band": 1,
                                       "x": 512,
                                       "y": 512
                                   },
                                   parallel=True)

            # add the time variable
            ds = ds.assign_coords(
                time=datetime.datetime.strptime(subdir, "%Y-%m-%d"))

            # add to the list
            list_ds.append(ds)
            pbar.update(1)

    # combine the datasets
    ds = xr.concat(list_ds, dim='time')
    ds.chunk({'time': 32, 'x': 32, 'y': 32})

    print('...loaded data, extracting now...')

    # check & convert crs
    if gdf.crs is None:
        raise ValueError("CRS not defined in the shapefile / geopackage")

    gdf = gdf.to_crs(ds.rio.crs)

    # prep
    if not os.path.exists(os.path.dirname(dir_out)):
        os.makedirs(os.path.dirname(dir_out), exist_ok=True)

    # extract the data
    with tqdm(total=len(gdf)) as pbar:
        for i in range(len(gdf)):

            # safety
            if gdf.geometry[i].is_empty:
                pbar.update(1)
                continue

            # work
            _ts._extract_timeseries(gdf,
                                    i,
                                    ds,
                                    dir_out,
                                    name_column,
                                    scale_factor,
                                    flag_value,
                                    flag_quality,
                                    cellbuffer_extend,
                                    extract_area=extract_area)
            pbar.update(1)

    print('...done.\n')

    pass


def plot_composite(
    directory: str,
    dates: list,
    fnpart_out: str,
    title: str,
    coarsen: int = 1,
    vmax=np.nan,
    value_factor: float = 1.0,
    max_factor: float = 1.0,
    save_composite: bool = False,
    extra_actions: str = '',
    lon_min: float = np.nan,
    lon_max: float = np.nan,
    lat_min: float = np.nan,
    lat_max: float = np.nan,
    fn_worldcover: str = '',
    extra_xlabel: str = '',
    exclude_lc_class: list = [],
    colourscale: str = 'viridis',
    export_stats: bool = False,
    stop_after_save: bool = False,
) -> None:
    """
    Plot a composite of the data in the given directory.
    
    :param directory: str, directory containing the data
    :param dates: list, list of dates to plot
    :param fnpart_out: str, filename part for the output file
    :param title: str, title for the plot
    :param coarsen: int, coarsening factor for the data
    :param vmax: float, maximum value for the plot
    :param value_factor: float, factor to apply to the values in the plot
    :param max_factor: float, factor to apply to the maximum value in the plot
    :param save_composite: bool, whether to save the composite data
    :param extra_actions: str, extra actions to perform (e.g. 'sum,mean,record,plot')
    :param lon_min: float, minimum longitude for the plot
    :param lon_max: float, maximum longitude for the plot
    :param lat_min: float, minimum latitude for the plot
    :param lat_max: float, maximum latitude for the plot
    :param fn_worldcover: str, filename of the world cover data to use
    :param extra_xlabel: str, extra label for the x-axis
    :param exclude_lc_class: list, list of land cover classes to exclude from the plot
    :param colourscale: str, colourscale to use for the plot
    :param export_stats: bool, whether to export statistics of the plot
    :param stop_after_save: bool, whether to stop after saving the dataset
    """

    # TODO add support for plotting the QC data

    print("\nStarting single composite plots...")

    # get list of all subdir dates available
    subdirs = _plot._get_all_subdirs(directory)
    subdir_dates = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in subdirs]

    if len(dates) > 0:
        dates_conv = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]
        dates_dirs = [d for d in dates_conv if d in subdir_dates]
    else:
        dates_dirs = subdir_dates

    print(f'Processing {len(dates_dirs)} composite(s):')

    for date in dates_dirs:
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n   - Processing {date_str}...")

        # get the files for this date
        files = [
            os.path.join(directory, date_str, f)
            for f in os.listdir(os.path.join(directory, date_str))
            if os.path.isfile(os.path.join(directory, date_str, f))
            and f.endswith('.tif')
        ]

        # load the data
        ds = xr.open_mfdataset(files,
                               combine='by_coords',
                               chunks={
                                   "band": 1,
                                   "x": 400,
                                   "y": 400
                               },
                               parallel=True)

        # add the time variable
        ds = ds.assign_coords(time=date).sel(band=1)

        # if save_composite is True, save the dataset to a zarr file
        if save_composite:
            data_dir_part = _plot._gen_data_dir_part(fnpart_out)
            fnpart_data = f"{os.path.basename(fnpart_out)}{date_str}.zarr"
            fnpart_data = fnpart_data.replace('__', '_')
        else:
            data_dir_part = ''
            fnpart_data = ''

        # plot the dataset
        plot_dataset(ds,
                     title=f"{title} for day {date_str}",
                     file_out=f"{fnpart_out}_{date_str}.png",
                     operator="only_plot",
                     coarsen=coarsen,
                     vmax=vmax,
                     value_factor=value_factor,
                     max_factor=max_factor,
                     data_output_dir=data_dir_part,
                     data_output_fn=fnpart_data,
                     extra_actions=extra_actions,
                     lon_min=lon_min,
                     lon_max=lon_max,
                     lat_min=lat_min,
                     lat_max=lat_max,
                     fn_worldcover=fn_worldcover,
                     extra_xlabel=extra_xlabel,
                     exclude_lc_class=exclude_lc_class,
                     colourscale=colourscale,
                     export_stats=export_stats,
                     stop_after_save=stop_after_save)

        ds.close()

    pass


def plot_dataset(
    dataset: xr.Dataset,
    title: str = '',
    file_out: str = '',
    operator: str = "mean",
    coarsen: int = 1,
    lon_min: float = np.nan,
    lon_max: float = np.nan,
    lat_min: float = np.nan,
    lat_max: float = np.nan,
    max_factor: float = 1.0,
    vmax: float = np.nan,
    value_factor: float = 1.0,
    data_output_dir: str = '',
    data_output_fn: str = '',
    continue_plot: bool = False,
    extra_actions: str = '',
    fn_worldcover: str = '',
    extra_xlabel: str = '',
    exclude_lc_class: list = [],
    colourscale: str = 'viridis',
    export_stats: bool = False,
    pixel_threshold: int = 25_000_000,
    plot_type: str = 'standard',
    stop_after_save: bool = False,
) -> None:
    """

    Plot a dataset with the given parameters.

    :param dataset: xr.Dataset, dataset to plot
    :param title: str, title for the plot
    :param file_out: str, filename for the output plot
    :param operator: str, operator to apply to the dataset (mean, sum, only_plot)
    :param coarsen: int, coarsening factor for the dataset
    :param lon_min: float, minimum longitude for the plot
    :param lon_max: float, maximum longitude for the plot
    :param lat_min: float, minimum latitude for the plot
    :param lat_max: float, maximum latitude for the plot
    :param max_factor: float, factor to apply to the maximum value in the plot
    :param vmax: float, maximum value for the plot
    :param value_factor: float, factor to apply to the values in the plot
    :param data_output_dir: str, directory to save the output dataset
    :param data_output_fn: str, filename for the output dataset
    :param continue_plot: bool, whether to continue from a previous plot
    :param extra_actions: str, extra actions to perform (e.g. 'sum,mean,record,plot')
    :param fn_worldcover: str, filename of the world cover data to use
    :param extra_xlabel: str, extra label for the x-axis
    :param exclude_lc_class: list, list of land cover classes to exclude from the plot
    :param colourscale: str, colourscale to use for the plot
    :param export_stats: bool, whether to export statistics of the plot
    :param pixel_threshold: int, threshold of pixels to switch between pcolormesh and imshow
    :param plot_type: str, type of plot to generate (standard, difference)
    :param stop_after_save: bool, whether to stop after saving the dataset

    operator keys, multiple can be used at once by joining them with the plus sign (+):
    - "mean": compute the mean of the dataset
    - "sum": compute the sum of the dataset
    - "only_plot": only plot the dataset without any aggregation
    - "record": output the dataset to a file without plotting

    """

    # TODO push the load part to each main function
    # TODO implement the "continue from fn_combined" option

    #
    # The core of the code was taken from
    # https://medium.com/@lubomirfranko/climate-data-visualisation-with-python-visualise-climate-data-using-cartopy-and-xarray-cf35a60ca8ee
    # with several generalizations and modifications
    #

    # safety checks
    if not file_out.endswith('.png'):
        raise ValueError(
            f"file_out must end with .png, but got {file_out} instead.")

    # settings
    fn_out = file_out.replace('__', '_')

    # safe data, if requested
    if data_output_fn != '' and not continue_plot:
        print("     Exporting the combined dataset...")

        # check if the output directory exists, if not, create it
        fn_combined = os.path.join(data_output_dir, data_output_fn)
        if not os.path.exists(os.path.dirname(fn_combined)):
            os.makedirs(os.path.dirname(fn_combined))

        # output the dataset to zarr
        if data_output_fn.endswith('.zarr'):
            compressor = zarr.codecs.BloscCodec(cname="zstd",
                                                clevel=3,
                                                shuffle="shuffle")
            enc = {x: {"compressor": compressor} for x in dataset}
            dataset.to_zarr(os.path.join(data_output_dir, data_output_fn),
                            mode='w',
                            encoding=enc,
                            consolidated=None)
        elif data_output_fn.endswith('.nc'):
            encoding = {var: {'dtype': 'float32'} for var in dataset.data_vars}
            dataset.to_netcdf(os.path.join(data_output_dir, data_output_fn),
                              mode='w',
                              encoding=encoding)
        elif data_output_fn.endswith('.tif') or data_output_fn.endswith(
                '.tiff'):
            # Save the dataset as a GeoTIFF file
            dataset.band_data.rio.to_raster(
                os.path.join(data_output_dir, data_output_fn),
                compress="ZSTD",
                predictor=2,
                bigtiff=True,
                tiled=True,
                # driver="COG",
                windowed=True)
        else:
            raise NotImplementedError(
                f"Data output format {data_output_fn} not supported. "
                "Supported formats are: .zarr, .nc, .tif, .tiff.")

        if stop_after_save:
            print("     Stopping after saving the dataset as requested.")
            return

    # coarsen the dataset
    # TODO safety checks with info msg for trimming
    # TODO there must be a more elegant way to do this
    print("     Running operator and coarsening...")
    # print("     - Resolution before coarsening: ", dataset.rio.resolution(),
    #       "with", dataset.sizes["x"] * dataset.sizes["y"],"number of cells")

    if operator == "mean":
        if "band" in dataset.dims:
            ds = dataset.sel(band=1).mean(dim='time', skipna=True).coarsen(
                x=coarsen, y=coarsen, boundary='trim').mean()
        else:
            ds = dataset.mean(dim='time',
                              skipna=True).coarsen(x=coarsen,
                                                   y=coarsen,
                                                   boundary='trim').max()
    elif operator == "sum":
        if "band" in dataset.dims:
            ds = dataset.sel(band=1).sum(dim='time', skipna=True).coarsen(
                x=coarsen, y=coarsen, boundary='trim').sum()
        else:
            ds = dataset.sum(dim='time',
                             skipna=True).coarsen(x=coarsen,
                                                  y=coarsen,
                                                  boundary='trim').max()
    elif operator == "only_plot":
        ds = dataset.coarsen(x=coarsen, y=coarsen, boundary='trim').max()
    else:
        raise ValueError(
            f"Operator {operator} not supported. Supported operators are: mean and sum."
        )

    # print("     - Resolution after coarsening: ", ds.rio.resolution(),
    #       "with", ds.sizes["x"] * ds.sizes["y"],"number of cells"  )

    # determine whether cropping is necessary
    if not np.isnan(lon_min) or not np.isnan(lon_max) or not np.isnan(
            lat_min) or not np.isnan(lat_max):
        cropped = True
    else:
        cropped = False

    # reprojection to EPSG:4326 if necessary
    if ds.rio.crs != "EPSG:4326":
        print("     Reprojecting...")

        # Get bounds in source CRS
        src_bounds = ds.rio.bounds()

        # Get the actual current pixel counts after coarsening
        n_pixels_x = ds.sizes["x"]
        n_pixels_y = ds.sizes["y"]

        # Transform bounds to EPSG:4326 to determine target extent
        from rasterio.warp import transform_bounds
        from rasterio.enums import Resampling

        target_bounds = transform_bounds(ds.rio.crs, "EPSG:4326",
                                         src_bounds[0], src_bounds[1],
                                         src_bounds[2], src_bounds[3])

        # If cropping, use the crop bounds to determine effective extent
        if cropped:
            eff_lon_min = lon_min if not np.isnan(
                lon_min) else target_bounds[0]
            eff_lon_max = lon_max if not np.isnan(
                lon_max) else target_bounds[2]
            eff_lat_min = lat_min if not np.isnan(
                lat_min) else target_bounds[1]
            eff_lat_max = lat_max if not np.isnan(
                lat_max) else target_bounds[3]
        else:
            eff_lon_min, eff_lat_min, eff_lon_max, eff_lat_max = target_bounds

        # Calculate extent in degrees
        extent_lon = abs(eff_lon_max - eff_lon_min)
        extent_lat = abs(eff_lat_max - eff_lat_min)

        # Calculate target resolution to maintain coarsened pixel density
        # This preserves the pixel count after coarsening
        target_res_lon = extent_lon / n_pixels_x
        target_res_lat = extent_lat / n_pixels_y
        target_res = min(target_res_lon, target_res_lat)

        # For output graphics at 12x6.75 inches at 150 DPI, we have ~1800x1012 pixels
        # Calculate minimum resolution needed for the display
        # This ensures we don't create unnecessarily large arrays
        if cropped:
            # Use actual plot dimensions (accounting for aspect ratio and margins)
            effective_plot_pixels = 1600  # Approximate after margins
            min_display_res_lon = extent_lon / effective_plot_pixels
            min_display_res_lat = extent_lat / effective_plot_pixels
            min_display_res = max(min_display_res_lon, min_display_res_lat)

            # Use the coarser of source-based resolution or display resolution
            # This prevents creating overly large arrays while preserving visible detail
            target_res = max(target_res, min_display_res)

        # Reproject with explicit resolution and bilinear resampling for smooth output
        ds = ds.rio.reproject("EPSG:4326",
                              resolution=target_res,
                              resampling=Resampling.bilinear)

    # print("     - Resolution after reprojection: ", ds.rio.resolution(),
    #       "with", ds.sizes["x"] * ds.sizes["y"],"number of cells"  )

    print("     Cropping, if necessary...")
    # Now, we will specify extent of our map in minimum/maximum longitude/latitude
    # Note that these values are specified in degrees of longitude and degrees of latitude
    # However, we can specify them in any crs that we want, but we need to provide appropriate
    # crs argument in ax.set_extent
    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees

    # Determine the dataset bounds
    if np.isnan(lon_min):
        long_min = ds.x.min().values
    else:
        long_min = lon_min
    if np.isnan(lon_max):
        long_max = ds.x.max().values
    else:
        long_max = lon_max
    if np.isnan(lat_min):
        lati_min = ds.y.min().values
    else:
        lati_min = lat_min
    if np.isnan(lat_max):
        lati_max = ds.y.max().values
    else:
        lati_max = lat_max

    if cropped:
        cropped_dataset = ds.sel(y=slice(lati_max, lati_min),
                                 x=slice(long_min, long_max))
    else:
        cropped_dataset = ds

    if value_factor != 1.0:
        # Apply value factor to the dataset
        cropped_dataset = cropped_dataset * value_factor

    if export_stats:
        stats_data = _plot.gen_stats(cropped_dataset,
                                     f"{fn_out[:-4]}_stats.txt")
    else:
        stats_data = {}

    # make the map
    # TODO add switch for whether actually to output
    print("     Making main plot:")
    _plot._make_main_plot(cropped_dataset,
                          cropped=cropped,
                          title=title,
                          file_out=fn_out,
                          vmax=vmax,
                          max_factor=max_factor,
                          lon_max=lon_max,
                          lon_min=lon_min,
                          lat_max=lat_max,
                          lat_min=lat_min,
                          extra_xlabel=extra_xlabel,
                          colourscale=colourscale,
                          pixel_threshold=pixel_threshold,
                          plot_type=plot_type,
                          stats_data=stats_data)

    # do the "side" plots
    if "plot" in extra_actions or "record" in extra_actions:

        # TODO add ability to _not_ plot when outputting data

        print("     Making longitude plots...")

        if data_output_dir == '':
            output_dir = _plot._gen_data_dir_part(fn_out)
        else:
            output_dir = data_output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if data_output_fn == '':
            output_fn = '.'.join(os.path.basename(fn_out).split('.')[:-1])
        else:
            output_fn = '.'.join(data_output_fn.split('.')[:-1])

        if "sum" in extra_actions:
            _plot._plot_longitude(
                cropped_dataset,
                title=title,
                fn_out=fn_out.replace('.png', '_longitude_sum'),
                fn_data_out=f"{output_dir}/{output_fn}_longitude_data_sum.csv",
                method='sum',
                output_data="record" in extra_actions,
                extra_xlabel=extra_xlabel)

        if "mean" in extra_actions:
            _plot._plot_longitude(
                cropped_dataset,
                title=title,
                fn_out=fn_out.replace('.png', '_longitude_mean'),
                fn_data_out=f"{output_dir}/{output_fn}_longitude_data_mean.csv",
                method='mean',
                output_data="record" in extra_actions,
                extra_xlabel=extra_xlabel)
        pass

        if fn_worldcover != '':
            print("     Plotting land cover distribution...")

            if 'record' in extra_actions:
                if data_output_dir == '':
                    output_dir = _plot._gen_data_dir_part(fn_out)
                else:
                    output_dir = data_output_dir
            else:
                output_dir = ''

            # TODO implement the world cover plotting
            _plot._plot_lc_percentage(cropped_dataset,
                                      fn_worldcover,
                                      title,
                                      fn_out.replace('.png',
                                                     '_lc_percentages.png'),
                                      output_dir,
                                      exclude_lc_class=exclude_lc_class)

            pass

        gc.collect()  # collect garbage to free memory

    pass


def plot_range(
    directory: str,
    dates: list,
    file_out: str,
    title: str,
    coarsen: int = 1,
    operator: str = "mean",
    low_mem: bool = False,
    tmp_dir: str = "tmp/",
    vmax=np.nan,
    output_data: bool = False,
    value_factor: float = 1.0,
    continue_plot: bool = False,
    extra_actions: str = '',  # 'sum,mean,record,plot'
    fn_worldcover: str = '',
    lon_min: float = np.nan,
    lon_max: float = np.nan,
    lat_min: float = np.nan,
    lat_max: float = np.nan,
    extra_xlabel: str = '',
    max_factor: float = np.nan,
    exclude_lc_class: list = [],
    colourscale: str = 'viridis',
    export_stats: bool = False,
    stop_after_save: bool = False,
) -> None:
    """
    Generate a plot for the given dataset over a range of dates.

    :param directory: str, directory containing the data
    :param dates: list, list of dates to plot
    :param file_out: str, filename for the output plot
    :param title: str, title for the plot
    :param coarsen: int, coarsening factor for the dataset
    :param operator: str, operator to apply to the dataset (mean, sum, only_plot)
    :param low_mem: bool, whether to use low memory mode
    :param tmp_dir: str, temporary directory for low memory mode
    :param vmax: float, maximum value for the plot
    :param output_data: bool, whether to output the combined dataset
    :param value_factor: float, factor to apply to the values in the plot
    :param continue_plot: bool, whether to continue from a previous plot
    :param extra_actions: str, extra actions to perform (e.g. 'sum,mean,record,plot')
    :param fn_worldcover: str, filename of the world cover data to use
    :param lon_min: float, minimum longitude for the plot
    :param lon_max: float, maximum longitude for the plot
    :param lat_min: float, minimum latitude for the plot
    :param lat_max: float, maximum latitude for the plot
    :param extra_xlabel: str, extra label for the x-axis
    :param max_factor: float, factor to apply to the maximum value in the plot
    :param exclude_lc_class: list, list
    :param colourscale: str, colourscale to use for the plot
    :param export_stats: bool, whether to export statistics of the plot
    :param stop_after_save: bool, whether to stop after saving the dataset
    """

    # TODO add support for plotting the QC data
    # TODO add plotting restart from fn_combined only

    if title != '':
        print(f"\nGenerating plot with title '{title}'...")
    else:
        print(f"\nGenerating plot to file {file_out}...")

    flag_only_plot = False
    if low_mem:
        # load the dataset in chunks
        ds = _plot._load_and_prework_dataset(directory,
                                             dates,
                                             tmp_dir,
                                             operator,
                                             skip_existing=continue_plot)
        flag_only_plot = True
    else:
        ds = _plot._load_whole_dataset(directory, dates)

    if flag_only_plot:
        operator_work = "only_plot"
    else:
        operator_work = operator

    if output_data:
        data_dir = _plot._gen_data_dir_part(file_out)
        data_output_fn = os.path.basename(file_out).replace('.png', '.zarr')
    else:
        data_dir = ''
        data_output_fn = ''

    plot_dataset(ds,
                 title=title,
                 file_out=file_out,
                 operator=operator_work,
                 coarsen=coarsen,
                 vmax=vmax,
                 value_factor=value_factor,
                 data_output_dir=data_dir,
                 data_output_fn=data_output_fn,
                 continue_plot=continue_plot,
                 extra_actions=extra_actions,
                 lon_min=lon_min,
                 lon_max=lon_max,
                 lat_min=lat_min,
                 lat_max=lat_max,
                 fn_worldcover=fn_worldcover,
                 extra_xlabel=extra_xlabel,
                 max_factor=max_factor,
                 exclude_lc_class=exclude_lc_class,
                 colourscale=colourscale,
                 export_stats=export_stats,
                 stop_after_save=stop_after_save)

    if low_mem:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    pass


if __name__ == "__main__":

    raise NotImplementedError(
        "This script is not meant to be run as a standalone script.")
