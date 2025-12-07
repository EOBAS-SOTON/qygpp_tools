#
# Helper routines for extracting time series information from the QY-GPP dataset.
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

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def _find_nearest_index(array, value) -> int:
    # from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1])
                    < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def _extract_timeseries(gdf: gpd.GeoDataFrame,
                        i: int,
                        ds: xr.Dataset,
                        dir_out: str,
                        name_column: str = '',
                        scale_factor: float = np.nan,
                        flag_value: bool = True,
                        flag_quality: bool = False,
                        cellbuffer_extend: int = 0,
                        extract_area: bool = False) -> None:

    point = gdf.geometry[i]

    # extract the data, deal w/o buffer
    if cellbuffer_extend > 0:
        x_pos = _find_nearest_index(ds.x.values, point.x)
        y_pos = _find_nearest_index(-ds.y.values, -point.y)

        # get the data
        data = ds.isel(x=slice(x_pos - cellbuffer_extend,
                               x_pos + cellbuffer_extend + 1),
                       y=slice(y_pos - cellbuffer_extend, y_pos +
                               cellbuffer_extend + 1)).mean(dim=['x', 'y'])

    else:
        data = ds.sel(x=point.x, y=point.y, method='nearest')

    # get the station name
    if name_column != '':
        name = gdf[name_column][i]
    else:
        name = f"point_{i:05}"

    if extract_area:
        data.to_netcdf(f'{dir_out}/time_series_area_{name}.nc', mode='w')

    # convert to output dataframe
    df = data.to_dataframe().reset_index(level=[0, 1])
    data.close()

    # column shenanigans
    if 'x' in df.columns and 'y' in df.columns:
        df.drop(columns=['x', 'y', "spatial_ref"], inplace=True)
    else:
        df.drop(columns=["spatial_ref"], inplace=True)
    list_out = []
    if flag_value:
        df_part = df[df["band"] == 1].drop(columns=["band"]).set_index("time")
        df_part.columns = ["data_value"]
        if not np.isnan(scale_factor):
            df_part["data_value"] = df_part["data_value"] * scale_factor
        list_out.append(df_part)
    if flag_quality:
        df_part = df[df["band"] == 2].drop(columns=["band"]).set_index("time")
        df_part.columns = ["quality_flag"]
        list_out.append(df_part)
    df_out = pd.concat(list_out, axis=1)

    # write to file
    df_out.to_csv(f'{dir_out}/time_series_{name}.csv', mode='w')


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. Use the main script instead."
    )
