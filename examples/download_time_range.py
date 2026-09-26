#
# Example: download all QY-GPP composites within a time range from the online
# repository (the S3 bucket behind the ESA Open Science Catalogue STAC entry
# https://opensciencedata.esa.int/products/gpp-sen4gpp/collection_v1).
#
# The files are stored in the directory structure expected by the analysis
# functions, i.e. <dir_target>/YYYY-MM-DD/QISCARF_GPP_*.tif
#

import datetime

from qygpp_tools import analysis

dir_target = "tmp/qygpp_test/"

# time range to download (inclusive), format "YYYY-MM-DD"
time_start = "2020-07-01"
time_end = "2020-07-31"

# MODIS tiles to download; set to None to download all tiles (~1.9 GB per date)
tiles = ["h18v03", "h18v04"]


def main():
    t_start = datetime.datetime.strptime(time_start, "%Y-%m-%d")
    t_end = datetime.datetime.strptime(time_end, "%Y-%m-%d")

    # the time-based query: all available composite dates within the range
    datums = [
        d for d in analysis.list_available_datums()
        if t_start <= datetime.datetime.strptime(d, "%Y-%m-%d") <= t_end
    ]
    print(
        f"Found {len(datums)} composite(s) between {time_start} and {time_end}:"
    )
    print(", ".join(datums))

    for datum in datums:
        analysis.download_datum_files(datum, dir_target, tiles=tiles)


if __name__ == "__main__":
    main()
