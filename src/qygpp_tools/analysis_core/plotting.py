#
# Helper routines for plotting the QY-GPP dataset.
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

import gc
import inspect
import os
import datetime
import math

import cartopy.crs as ccrs
import cartopy.feature as cf
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio as rio
from tqdm import tqdm
import xarray as xr

# set the default matplotlib backend to Agg (weird plotting errors otherwise)
import matplotlib

matplotlib.use('Agg')

#TODO make this unnecessary
import warnings

warnings.filterwarnings("ignore")

data_output_part = "data_save/"

labels_worldcover = {
    10: "Tree Cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and Ice",
    80: "Permanent Water Bodies",
    90: "Herbaceous Wetlands",
    95: "Mangroves",
    100: "Moss and Lichens",
}


def _plot_lc_percentage(ds: xr.Dataset,
                        fn_worldcover: str,
                        title: str = '',
                        fn_out: str = '',
                        output_data_dir: str = '',
                        exclude_lc_class: list = []) -> None:

    # TODO check and change behaviour for non-Worldcover datasets

    ds_lc = xr.open_mfdataset(fn_worldcover,
                              decode_times=False,
                              chunks={
                                  "x": 1600,
                                  "y": 1280
                              },
                              parallel=False)

    # ds_lc = ds_lc.rename(name_dict={'longitude': 'x', 'latitude': 'y'})

    # reproject and resample
    print('     - Reprojecting land cover data to match GPP data...')
    matched_lc = ds_lc.rio.reproject_match(ds,
                                           resampling=rio.enums.Resampling(6))
    # matched_lc = matched_lc.astype('int32')  # makes it smaller in memory

    # print(matched_lc.band_data.values)
    # print(ds.band_data.values)
    # exit()

    # squash the land cover classes to the GPP dataset
    print('   - Squashing land cover classes to GPP data...')
    matched_lc = np.squeeze(matched_lc.band_data.values)
    gpp = np.squeeze(ds.band_data.values)

    # get the unique land cover classes
    print('   - Getting unique land cover classes...')
    lc_classes = np.unique(matched_lc)

    data_lc = {}
    for value in lc_classes:
        if np.isnan(value):
            continue

        if labels_worldcover[value] in exclude_lc_class:
            continue

        data_lc[value] = [
            np.nansum(gpp[matched_lc == value]), labels_worldcover[value]
        ]

    # create dataframe
    print('   - Creating dataframe...')
    df = pd.DataFrame.from_dict(data_lc,
                                orient='index',
                                columns=['GPP Sum', 'Land Cover Class'])

    # add percentage of GPP for each land cover class
    df['Percentage'] = round((df['GPP Sum'] / df['GPP Sum'].sum()) * 100, 1)

    if output_data_dir != '':

        fn = os.path.join(
            output_data_dir,
            os.path.basename(fn_out.replace('.png', '_data.csv')))

        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

        # output the data to a csv file
        df_out = df.reset_index().rename(columns={'index': 'Land Cover Class'})
        df_out.to_csv(fn)

    # plot
    print('   - Plotting...')
    if title == '':
        display_title = ""
    else:
        display_title = f"Percentage GPP per Landcover of<br>{title}"
    fig = px.pie(df,
                 values='Percentage',
                 names='Land Cover Class',
                 title=display_title,
                 color_discrete_sequence=px.colors.qualitative.Vivid)

    fig.write_image(fn_out)

    gc.collect()

    pass


def _plot_longitude(ds: xr.Dataset,
                    title: str = '',
                    fn_out: str = '',
                    fn_data_out: str = '',
                    method: str = '',
                    output_data: bool = False,
                    extra_xlabel: str = '') -> None:

    # generate the data
    if method == 'sum':
        df_longitude = ds.sum(dim='x', skipna=True)
    elif method == 'mean':
        df_longitude = ds.mean(dim='x', skipna=True)
        df_longitude = df_longitude.fillna(0.0)
    else:
        raise ValueError(f"Method {method} not supported. "
                         "Supported methods are: sum and mean.")

    if extra_xlabel != '':
        xlabel = f'GPP {method.capitalize()} (gC/m²/{extra_xlabel})'
    else:
        xlabel = f'GPP {method.capitalize()} (gC/m²)'

    # plot as a line graph with latitude on the y-axis and the value on the x-axis
    plt.figure(figsize=(6.75, 12), dpi=150)
    plt.plot(df_longitude.band_data, df_longitude.y)  # , color='blue')
    if title != '':
        plt.title(f"GPP {method.capitalize()} per Longitude\n{title}")
    plt.ylabel('Latitude')
    plt.xlabel(xlabel)

    # check if output location exists
    path_out = os.path.dirname(fn_out)
    if path_out != '':
        if not os.path.exists(path_out):
            os.makedirs(path_out)
    plt.savefig(f'{fn_out}.png', dpi=300, bbox_inches='tight')
    plt.close()
    gc.collect()

    if output_data:
        # output the data to a csv file
        df = df_longitude.to_dataframe().reset_index()
        df = df.drop(columns=['time', 'band', 'spatial_ref'],
                     errors='ignore').dropna()
        df.to_csv(fn_data_out, index=False)


def _gen_data_dir_part(file_out: str) -> str:
    """
    Generate the directory name for data output based on the file_out path.
    """
    path_out = (f"{os.path.dirname(file_out)}/{data_output_part}/")

    return path_out


def _extract_from_ds(val):
    if isinstance(val, xr.DataArray):
        if "band_data" in val.dims:
            print(xr.Dataset.to_array(val.band_data))
            exit()

            return val.band_data.values[0]
        elif "band" in val.dims:
            print(val.sel(band=1)[0])
            exit()
            return val.sel(band=1).values[0]
        else:
            raise ValueError(
                f"DataArray {val} does not have 'band_data' or 'band' dimension.\n"
                f"Info: {val.dims}, {val.coords}, {val.attrs}\n"
                f"{val}")
    else:
        print("else")
        print(val(), type(val()))
        # print(val.values)
        # print(val.band_data[0])
        # print(val['band_data'].values)
        exit()

        if inspect.isfunction(val):
            # if val is a function, call it
            val = val()
        elif isinstance(val, float) or isinstance(val, int):
            return val
        else:
            raise ValueError(
                f"Value {val} is not a DataArray or a number. Type: {type(val)}"
            )


def _make_main_plot(
    dataset: xr.Dataset,
    cropped: bool = False,
    vmax: float = np.nan,
    max_factor: float = 1.0,
    title: str = '',
    file_out: str = '',
    lon_min: float = np.nan,
    lon_max: float = np.nan,
    lat_min: float = np.nan,
    lat_max: float = np.nan,
    extra_xlabel: str = '',
    colourscale: str = 'viridis',
    pixel_threshold: int = 25_000_000,
    plot_type: str = 'standard',
    stats_data: dict = {},
) -> None:

    # the dataset needs to be loaded into memory at one point here
    # make certain that it is in float32 format (only data variables, not coordinates)
    for var in dataset.data_vars:
        if dataset[var].dtype != np.float32:
            dataset[var] = dataset[var].astype('float32')
    gc.collect()

    # Depending on crop or not, get the main statistics
    # Compute min, mean, max in a single pass using dask delayed computation
    print("     Statistics...")
    if "band" in dataset.dims:
        ds_stats = dataset.sel(band=1)
    else:
        ds_stats = dataset

    # Create all reductions, then compute together in one pass
    if stats_data == {}:
        stats = xr.Dataset({
            'min': ds_stats.band_data.min(),
            'mean': ds_stats.band_data.mean(),
            'max': ds_stats.band_data.max()
        })

        # Compute with progress bar
        with ProgressBar():
            stats = stats.compute()
    else:
        stats = stats_data

    min_val = float(stats['min'].values)
    mean_val = float(stats['mean'].values)
    max_val = float(stats['max'].values)

    print("     Plotting...")
    # First we specify Coordinate Refference System for Map Projection
    # We will use Mercator, which is a cylindrical, conformal projection.
    # It has bery large distortion at high latitudes, cannot
    # fully reach the polar regions.
    # projection = ccrs.Mercator()

    if cropped:
        projection = ccrs.Miller(central_longitude=((lon_max - lon_min) / 2))
    else:
        # Should look nicer for GPP (little near the poles)
        projection = ccrs.Robinson(central_longitude=0)

    # Specify CRS, that will be used to tell the code, where should our data be plotted
    crs = ccrs.PlateCarree()

    # Now we will create axes object having specific projection
    plt.figure(figsize=(12, 6.75), dpi=150)
    ax = plt.axes(projection=projection, frameon=True)

    # Draw gridlines in degrees over Mercator map
    gl = ax.gridlines(crs=crs,
                      draw_labels=True,
                      linewidth=.6,
                      color='gray',
                      alpha=0.5,
                      linestyle='-.')
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # To plot borders and coastlines, we can use cartopy feature
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)

    ##### WE ADDED THESE LINES #####
    if extra_xlabel != '':
        xlabel = f'Gross Primary Productivity (gC/m²/{extra_xlabel})'
    else:
        xlabel = 'Gross Primary Productivity (gC/m²)'

    if plot_type == 'standard':
        if np.isnan(vmax):
            vmax_plot = math.ceil(max(max_val * max_factor, mean_val))
        else:
            vmax_plot = math.ceil(max(vmax, 20.0))
    elif plot_type == 'difference':
        if np.isnan(vmax):
            vmax_plot = math.ceil(
                max(abs(min_val) * max_factor,
                    abs(max_val) * max_factor))
            vmin_plot = -vmax_plot
        else:
            vmax_plot = math.ceil(max(vmax, 20.0))
            vmin_plot = -vmax_plot
    else:
        raise NotImplementedError(
            f"Plot type {plot_type} not supported. Supported types are: standard and difference."
        )

    # set values to nan if they are below 0.01
    # print("\nds_cropped", type(cropped_dataset), "\n", cropped_dataset)
    if plot_type == 'standard':
        dataset = dataset.where(dataset > 0.0001)

    # Determine which plotting method to use based on pixel count
    # Cutoff: 25 million pixels (≈600MB per plot with pcolormesh)
    # This allows high-quality pcolormesh for regional/continental plots
    # while using memory-efficient imshow for very large global plots
    data_shape = dataset.band_data.shape
    total_pixels = data_shape[-1] * data_shape[-2]  # width × height
    use_pcolormesh = total_pixels <= pixel_threshold

    # Create normalization for smooth color gradients
    # For difference plots, use symmetric range centered on 0
    if plot_type == 'difference':
        norm = Normalize(vmin=vmin_plot, vmax=vmax_plot, clip=True)
        plot_cmap = cm.get_cmap('coolwarm')
    else:
        norm = Normalize(vmin=0.0001, vmax=vmax_plot, clip=True)
        plot_cmap = cm.get_cmap(colourscale.lower())

    if use_pcolormesh:
        print(f"     - Using pcolormesh for plotting, {total_pixels:n} pixels")
        # Use pcolormesh for better high-resolution rendering
        # pcolormesh preserves all pixels without regridding/resampling

        # Get coordinate arrays
        x_coords = dataset.x.values
        y_coords = dataset.y.values
        # Squeeze to remove any single-element dimensions (e.g., band)
        data_values = np.squeeze(dataset.band_data.values)

        # Plot with pcolormesh - this preserves every pixel
        mesh = ax.pcolormesh(x_coords,
                             y_coords,
                             data_values,
                             transform=ccrs.PlateCarree(),
                             cmap=plot_cmap,
                             norm=norm,
                             shading='auto',
                             rasterized=True)

        # Free memory from intermediate arrays immediately after plotting
        del x_coords, y_coords, data_values, norm
        gc.collect()
    else:
        print(f"     - Using imshow for plotting, {total_pixels:n} pixels")
        # Use imshow for very large datasets to conserve memory
        # This is more memory-efficient for global-scale plots
        cbar_kwargs = {
            'orientation': 'horizontal',
            'shrink': 0.6,
            "pad": .05,
            'aspect': 40,
            'label': xlabel,
        }

        # Set vmin based on plot type
        if plot_type == 'difference':
            imshow_vmin = vmin_plot
        else:
            imshow_vmin = 0.0001

        dataset.band_data.plot.imshow(ax=ax,
                                      transform=ccrs.PlateCarree(),
                                      cmap=plot_cmap,
                                      cbar_kwargs=cbar_kwargs,
                                      vmin=imshow_vmin,
                                      vmax=vmax_plot,
                                      levels=21)
        gc.collect()

    # Calculate adaptive colorbar width based on actual data dimensions
    aspect_ratio = 1.0  # default
    if len(data_shape) >= 2:
        # Get width (x) and height (y) of the data
        width = data_shape[-1]  # x dimension
        height = data_shape[-2]  # y dimension
        aspect_ratio = width / height if height > 0 else 1.0

        # Adjust shrink and aspect based on map aspect ratio
        # Square/tall maps get smaller colorbar, wide maps get wider colorbar
        shrink_factor = min(0.75, max(0.5, 0.3 + 0.15 * aspect_ratio))
        cbar_aspect = max(25, min(50, 30 * aspect_ratio))
    else:
        # Default values if dimensions are unexpected
        shrink_factor = 0.6
        cbar_aspect = 40

    # Add colorbar manually with nice rounded tick values and adaptive sizing
    cbar = plt.colorbar(mesh,
                        ax=ax,
                        orientation='horizontal',
                        shrink=shrink_factor,
                        pad=.05,
                        aspect=cbar_aspect,
                        anchor=(0.5, 0.5),
                        label=xlabel)
    # Use MaxNLocator to get nice round numbers with fewer ticks
    cbar.locator = MaxNLocator(nbins=8, integer=False)
    cbar.update_ticks()
    ################################

    if cropped:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)

    if title != '':
        plt.title(f"{title}\nmean: {mean_val:.1f}, max: {max_val:.1f} gC/m²", )
    else:
        plt.title("")
    # plt.title(title)
    # plt.show()

    # check if output location exists
    path_out = os.path.dirname(file_out)
    if path_out != '':
        if not os.path.exists(path_out):
            os.makedirs(path_out)
    plt.savefig(file_out, dpi=300, bbox_inches='tight')

    plt.close()

    # Free remaining plot objects and dataset
    del mesh, cbar, ax, dataset
    gc.collect()


def _get_subdir_list(subdirs: list,
                     time_start: str = '',
                     time_end: str = '') -> list[str]:

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
    subdir_range = [
        d for d in subdirs
        if datetime.datetime.strptime(d, "%Y-%m-%d") >= t_start
        and datetime.datetime.strptime(d, "%Y-%m-%d") <= t_end
    ]

    return subdir_range


def _get_subdir_ranges(subdirs: list, dates: list) -> list:

    # Get list of files to work on
    if isinstance(dates[0], list):
        # get the subdirectories in the time range
        subdir_range = []
        for date in dates:
            subdir_range += _get_subdir_list(subdirs, date[0], date[1])
    else:
        subdir_range = _get_subdir_list(subdirs, dates[0], dates[1])

    return subdir_range


def _get_all_subdirs(directory: str) -> list[str]:
    # get all dates in the directory
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

    return subdirs


def _load_whole_dataset(directory: str, dates: list) -> xr.Dataset:

    subdirs = _get_all_subdirs(directory)

    subdir_range = _get_subdir_ranges(subdirs, dates)

    # print(subdir_range)

    list_ds = []
    print(f"   - Loading {len(subdir_range)} subdirectories / times...")
    with tqdm(total=len(subdir_range)) as pbar:
        for subdir in subdir_range:
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

    return ds


def _load_and_prework_dataset(directory: str,
                              dates: list,
                              tmp_dir: str,
                              operator: str,
                              skip_existing: bool = False) -> xr.Dataset:

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    subdirs = _get_all_subdirs(directory)
    subdir_range = _get_subdir_ranges(subdirs, dates)

    # now get list of files in the first subdir
    files = [
        f for f in os.listdir(os.path.join(directory, subdir_range[0]))
        if os.path.isfile(os.path.join(directory, subdir_range[0], f))
        and f.endswith('.tif')
    ]

    # now process each tile across all subdirs
    list_fn_out = []
    print('   - preparing tiles...')
    with tqdm(total=len(files)) as pbar:
        for f in files:

            # prep
            parts = f.split("_")
            beginning = '_'.join(parts[:2])
            ending = '_'.join(parts[3:])
            fn_out = os.path.join(tmp_dir, ending.replace('.tif', '.nc'))

            if skip_existing and os.path.exists(fn_out):
                list_fn_out.append(fn_out)
                pbar.update(1)
                continue

            # load the data
            list_ds = []
            crs = None
            for subdir in subdir_range:

                # create filename
                fn = f"{beginning}_{subdir.replace('-', '')}_{ending}"

                # get the data
                ds = xr.open_mfdataset([os.path.join(directory, subdir, fn)],
                                       combine='by_coords',
                                       chunks={
                                           "band": 1,
                                           "x": 800,
                                           "y": 800
                                       },
                                       parallel=True)

                if crs is None:
                    crs = ds.rio.crs

                # add the time variable
                ds = ds.assign_coords(
                    time=datetime.datetime.strptime(subdir, "%Y-%m-%d"))

                # add to the list
                list_ds.append(ds)

            # combine the datasets
            ds = xr.concat(list_ds, dim='time')

            # compute the operator
            if operator == "mean":
                ds = ds.sel(band=1).mean(dim='time', skipna=True)
            elif operator == "sum":
                ds = ds.sel(band=1).sum(dim='time', skipna=True)
            else:
                raise ValueError(f"Operator {operator} not supported. "
                                 "Supported operators are: mean and sum.")

            ds.rio.write_crs(crs)
            list_fn_out.append(fn_out)

            # Specify encoding to save as float32
            encoding = {var: {'dtype': 'float32'} for var in ds.data_vars}
            ds.to_netcdf(fn_out, encoding=encoding)
            ds.close()
            pbar.update(1)

    print("   - Loading processed dataset...")
    ds = xr.open_mfdataset(list_fn_out,
                           combine='by_coords',
                           chunks={
                               "band": 1,
                               "x": 400,
                               "y": 400
                           },
                           parallel=False)

    return ds


def gen_stats(ds: xr.Dataset, fn_out: str):

    print("   - Generating statistics...")

    # Create all reductions lazily, then compute together in one pass
    stats_lazy = xr.Dataset({
        'mean': ds.band_data.mean(dim=["x", "y"]),
        'min': ds.band_data.min(dim=["x", "y"]),
        'max': ds.band_data.max(dim=["x", "y"]),
        'sum': ds.band_data.sum(dim=["x", "y"]),
        'std': ds.band_data.std(dim=["x", "y"]),
        'var': ds.band_data.var(dim=["x", "y"]),
    })

    # Compute with progress bar
    with ProgressBar():
        stats = stats_lazy.compute()

    # Save statistics to file
    with open(fn_out, "w") as f:
        for key in ['mean', 'min', 'max', 'sum', 'std', 'var']:
            f.write(f"{key}: {stats[key].values}\n")

    return stats


if __name__ == "__main__":

    raise NotImplementedError(
        "This script is not meant to be run as a standalone script.")
