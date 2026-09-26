#
# Example: query the QY-GPP STAC catalogue directly by time range and area, then
# download the matching tiles.
#
# The catalogue (https://opensciencedata.esa.int/products/gpp-sen4gpp/collection_v1)
# is a static STAC catalogue, i.e. there is no /search endpoint. The query is
# therefore done client-side by walking the catalogue:
#   catalog -> one collection per 8-day composite -> one item per MODIS tile
#
# Requires pystac in addition to the qygpp_tools dependencies.
#
# The files are stored in the directory structure expected by the analysis
# functions, i.e. <dir_target>/YYYY-MM-DD/QISCARF_GPP_*.tif
#

import datetime
import functools
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pystac
from pyproj import Transformer
from pystac.extensions.projection import ProjectionExtension
from shapely.geometry import box
from shapely.ops import transform
from tqdm import tqdm

url_catalog = "https://s3.waw4-1.cloudferro.com/EarthCODE/Catalogs/sen4gpp/catalog.json"

dir_target = "tmp/qygpp_test/"

# time range to query (inclusive), format "YYYY-MM-DD"; every composite whose
# 8-day period overlaps the range is selected
time_start = "2020-07-01"
time_end = "2020-07-31"

# area to query as [lon_min, lat_min, lon_max, lat_max] (WGS84); here the
# Southampton / New Forest area
bbox = [-2.0, 50.7, -1.0, 51.2]

# number of parallel requests when reading the catalogue
n_workers = 16


@functools.lru_cache
def get_transformer(wkt: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", wkt, always_xy=True)


def intersects_tile(item: pystac.Item, area) -> bool:
    # The lon/lat item geometry is unusable for tiles at the edge of the
    # sinusoidal grid (it wraps around the antimeridian and spans most of the
    # globe). Instead, the query area is projected into the tile's sinusoidal
    # CRS and tested against the exact tile extent (proj:bbox).
    proj = ProjectionExtension.ext(item.assets["tile"])
    area_tile = transform(
        get_transformer(proj.wkt2).transform, area.segmentize(0.1))
    return box(*proj.bbox).intersects(area_tile)


def query_catalog():
    t_start = datetime.datetime.strptime(
        time_start, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    t_end = datetime.datetime.strptime(time_end, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc)
    area = box(*bbox)

    catalog = pystac.Catalog.from_file(url_catalog)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        # temporal filter on the per-composite collections
        hrefs = [
            link.get_absolute_href() for link in catalog.get_child_links()
        ]
        collections = list(
            tqdm(
                pool.map(pystac.Collection.from_file, hrefs),
                total=len(hrefs),
                desc="Reading collections",
            ))
        collections = [
            c for c in collections
            if c.extent.temporal.intervals[0][0] <= t_end
            and c.extent.temporal.intervals[0][1] >= t_start
        ]
        print(
            f"{len(collections)} composite(s) overlap {time_start} - {time_end}"
        )

        # spatial filter on the per-tile items
        items = []
        for collection in collections:
            hrefs = [
                link.get_absolute_href()
                for link in collection.get_item_links()
            ]
            items_all = tqdm(
                pool.map(pystac.Item.from_file, hrefs),
                total=len(hrefs),
                desc=f"Reading items {collection.id}",
            )
            items += [i for i in items_all if intersects_tile(i, area)]

    return items


def download_items(items):
    for item in tqdm(items, desc="Downloading"):
        href = item.assets["tile"].href
        dir_datum = os.path.join(dir_target,
                                 item.datetime.strftime("%Y-%m-%d"))
        os.makedirs(dir_datum, exist_ok=True)
        with urllib.request.urlopen(href) as response, open(
                os.path.join(dir_datum, os.path.basename(href)), "wb") as f:
            shutil.copyfileobj(response, f)


def main():
    items = query_catalog()
    print(f"Found {len(items)} tile(s):")
    for item in items:
        print(" ", item.id, item.properties["title"])
    download_items(items)


if __name__ == "__main__":
    main()
