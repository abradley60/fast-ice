# Module for loading from various sources and returning xarrays
from pystac import Item, ItemCollection
import rioxarray
import odc.geo.xr  # registers .odc. accessor on xarrays
from odc.geo.xr import assign_crs
from odc.geo import CRS, GeoBox, BoundingBox
import xarray as xr
import numpy as np
import geopandas as gpd
import stac_geoparquet
from odc.stac import load
import os
from pystac_client import Client
from typing import Literal, get_args, Any, Protocol, runtime_checkable
from datetime import datetime
from shapely import Geometry
from shapely.geometry import shape

from fast_ice.utils import convert_linear_to_db_xr

METHOD = Literal["geoparquet", "api"]
SAR_UNIT = Literal["linear", "db"]


# Create GeoInterface type to support type checking
@runtime_checkable
class GeoInterface(Protocol):
    @property
    def __geo_interface__(self) -> dict: ...


def _validate_and_standardise_sentinel1(da: xr.DataArray) -> xr.DataArray:
    """Format the supplied xarray in preparation for the norm prod library.
    If the xarray.DataArray has a coordinate reference system, ensure it's listed in the coordinates.
    If xarray.DataArray specifies linear units, convert to decibels.

    Parameters
    ----------
    da : xr.DataArray
        xarray.DataArray containing Sentinel-1 data and unit attribute

    Returns
    -------
    xr.DataArray
        The input xarray.DataArray in decibel units

    Raises
    ------
    ValueError
        If no coordinate reference system is found.
    ValueError
        If a unit other than "db" or "linear" is supplied.
    """

    # Check that CRS exists, and update CRS in coords table to match ODC convention
    if da.rio.crs is None:
        raise ValueError("No CRS found from file.")
    da = assign_crs(da, crs=da.rio.crs)

    # Check unit attribute is valid
    da_unit = da.attrs["unit"]
    allowed_units = get_args(SAR_UNIT)
    if da_unit not in allowed_units:
        raise ValueError(
            f"Unexpected unit: {da_unit!r}. Allowed units are: {allowed_units}"
        )

    # If linear, convert to decibels.
    if da.attrs["unit"] == "linear":
        da = convert_linear_to_db_xr(da)
        da.attrs["unit"] = "db"

    return da


def _load_from_file(filename: str | os.PathLike) -> xr.DataArray:
    """Light wrapper to load using rioxarray and drop the band dimension"""

    result = rioxarray.open_rasterio(filename)

    if not isinstance(result, xr.DataArray):
        raise TypeError(f"Expected a DataArray, got {type(result)}")
    return result.squeeze("band", drop=True)


def load_sentinel_1_from_file(
    filename: str | os.PathLike,
    band: str,
    band_unit: SAR_UNIT | None,
    timestamp: datetime | None,
) -> xr.DataArray:
    """Load Sentinel-1 data from a file into an xarray.DataArray and
    attach required metadata (band, band unit, and timestamp)

    Parameters
    ----------
    filename : str | os.PathLike
        Path to the file to load.
    band : str
        Name of the band (e.g. hh_gamma0).
    band_unit : SAR_UNIT | None
        Unit of the band (either db, linear, or None if part of file metadata).
    timestamp : datetime | None
        Timestamp of the file (None if part of the file metadata).

    Returns
    -------
    xr.DataArray
        Loaded Sentinel-1 observation with appropriate metadata.

    Raises
    ------
    ValueError
        If no timestamp is found or provided.
    ValueError
        If no band unit is found or provided.
    """

    da = _load_from_file(filename)

    # Name the array using the band's name
    da = da.rename(band)

    # Set time coordinate
    if "time" not in da.coords:
        if timestamp is None:
            raise ValueError("datetime must be provided if not already present in file")
        da.coords["time"] = timestamp

    # Set unit attribute
    if "unit" not in da.attrs:
        if band_unit is None:
            raise ValueError("unit must be provided if not already present in file")
        da.attrs["unit"] = band_unit

    da = _validate_and_standardise_sentinel1(da)

    return da


def _build_cql2_filter(filters: dict | None) -> dict | None:
    """Utility function to allow the user to construct a filter dictionary for
    odc-stac.load from a simple dictionary of filter property and value

    Parameters
    ----------
    filters : dict | None
        A dictionary of key-value pairs to use for filtering, assuming equality
        e.g. {'sat:relative_orbit': 40} -> 'sat:relative_orbit' == 40

    Returns
    -------
    dict | None
        A dictionary following the CQL2 JSON standard, suitable for odc-stac.load
    """

    if not filters:
        return None

    cql2_filters = [
        {
            "op": "=",
            "args": [{"property": key}, value],
        }
        for key, value in filters.items()
    ]
    return (
        {"op": "and", "args": cql2_filters}
        if len(cql2_filters) > 1
        else cql2_filters[0]
    )


def _stac_items_from_api(
    stac_endpoint: str,
    collection: str,
    start_time: datetime | str,
    end_time: datetime | str,
    intersects: GeoInterface,
    filters: dict | None = None,  # key-value pairs with the assumed operator "="
) -> ItemCollection:
    """Retrieve STAC items from a STAC API using pystac-client

    Parameters
    ----------
    stac_endpoint : str
        URL to search for STAC items (e.g. https://explorer.dev.dea.ga.gov.au/stac).
    collection : str
        STAC collection to search (e.g. ga_s1_nrb_iw_hh_1).
    start_time : datetime | str
        Time to search from (e.g. "2021-01-01" or datetime(2021, 01, 01)).
    end_time : datetime | str
        Time to search to (e.g. "2021-01-31" or datetime(2021, 01, 31)).
    intersects : GeoInterface
        The geometry to intersect with, must have the "__geointerface__" property.
    filters : dict | None, optional
        Any filters to apply to the STAC query, by default None.
        Only equality (=) is supported for filters.
        e.g. {'sat:relative_orbit': 40} -> 'sat:relative_orbit' = 40
        e.g. {'eo:platform': 'Sentinel-1A'} -> 'eo:platform' = 'Sentinel-1A'

    Returns
    -------
    ItemCollection
        A list of STAC items that can be passed to a loader.
    """

    stac_client = Client.open(stac_endpoint)

    cql2_filter = _build_cql2_filter(filters)

    # Convert datetimes to strings for use in query
    if isinstance(start_time, datetime):
        start_time = start_time.isoformat()
    if isinstance(end_time, datetime):
        end_time = end_time.isoformat()

    search_kwargs = {
        "collections": [collection],
        "datetime": f"{start_time}/{end_time}",
        "intersects": intersects,
    }
    if cql2_filter is not None:
        search_kwargs["filter"] = cql2_filter

    return stac_client.search(**search_kwargs).item_collection()


def _stac_items_from_geoparquet(
    geoparquet_file: str | os.PathLike,
    collection: str,
    start_time: datetime,
    end_time: datetime,
    intersects: Geometry,
    filters: dict | None,
) -> ItemCollection:
    """Retrieve STAC items from a STAC GeoParquet using geopandas

    Parameters
    ----------
    geoparquet_file : str
        File to search for STAC items (e.g. ga_s1_nrb_ew_hh_hv_1_v2.parquet).
    collection : str
        STAC collection to search (e.g. ga_s1_nrb_ew_hh_hv_1).
    start_time : datetime | str
        Time to search from (e.g. "2021-01-01" or datetime(2021, 01, 01)).
    end_time : datetime | str
        Time to search to (e.g. "2021-01-31" or datetime(2021, 01, 31)).
    intersects : GeoInterface
        The geometry to intersect with, must have the "__geointerface__" property.
    filters : dict | None, optional
        Any filters to apply to the STAC query, by default None.
        Only equality (=) is supported for filters.
        e.g. {'sat:relative_orbit': 40} -> 'sat:relative_orbit' = 40
        e.g. {'eo:platform': 'Sentinel-1A'} -> 'eo:platform' = 'Sentinel-1A'

    Returns
    -------
    ItemCollection
        A list of STAC items that can be passed to a loader.
    """

    stac_index = gpd.read_parquet(geoparquet_file)

    # filter on collection
    items_gdf = stac_index[stac_index["collection"] == collection]
    # filter on datetime
    items_gdf = items_gdf[
        (items_gdf.datetime >= start_time.isoformat())
        & (items_gdf.datetime <= end_time.isoformat())
    ]
    # filter on geometry
    items_gdf = items_gdf[items_gdf.intersects(intersects)]

    # apply other filters
    for key, value in (filters or {}).items():
        items_gdf = items_gdf[items_gdf[key] == value]

    sorted_items_gdf = items_gdf.copy().sort_values("datetime").reset_index(drop=True)

    return stac_geoparquet.to_item_collection(sorted_items_gdf)


def query_sentinel_1_stac(
    collection: str,
    start_time: datetime,
    end_time: datetime,
    intersects: GeoInterface,
    filters: dict | None = None,  # key-value pairs with the assumed operator "="
    *,
    method: METHOD = "api",
    api_endpoint: str | None = None,
    geoparquet_file: str | os.PathLike | None = None,
) -> ItemCollection:
    """Retrieve STAC items from a STAC API or STAC GeoParquet file

    Parameters
    ----------
    collection : str
        STAC collection to search (e.g. ga_s1_nrb_ew_hh_hv_1).
    start_time : datetime | str
        Time to search from (e.g. "2021-01-01" or datetime(2021, 01, 01)).
    end_time : datetime | str
        Time to search to (e.g. "2021-01-31" or datetime(2021, 01, 31)).
    intersects : GeoInterface
        The geometry to intersect with, must have the '__geointerface__' property.
    filters : dict | None, optional
        Any filters to apply to the STAC query, by default None.
        Only equality (=) is supported for filters.
        e.g. {'sat:relative_orbit': 40} -> 'sat:relative_orbit' = 40
        e.g. {'eo:platform': 'Sentinel-1A'} -> 'eo:platform' = 'Sentinel-1A'
    method : METHOD, optional
        Method to use, either 'api' or 'geoparquet', by default 'api'.
    api_endpoint : str | None, optional
        URL to search for STAC items (e.g. https://explorer.dev.dea.ga.gov.au/stac), by default None
    geoparquet_file : str | os.PathLike | None, optional
       File to search for STAC items (e.g. ga_s1_nrb_ew_hh_hv_1_v2.parquet), by default None

    Returns
    -------
    ItemCollection
        A list of STAC items that can be passed to a loader.

    Raises
    ------
    ValueError
        If no STAC API endpoint is supplied when using method='api'
    ValueError
        If no STAC GeoParquet file is supplied when using method='geoparquet'
    ValueError
        If an unrecognised method is supplied. Allowed methods are 'api' and 'geoparquet'
    ValueError
        If no STAC items are returned from the search.
    """

    if method == "api":
        if api_endpoint is None:
            raise ValueError("api_endpoint required when method='api'")

        stac_items = _stac_items_from_api(
            api_endpoint,
            collection,
            start_time,
            end_time,
            intersects,
            filters=filters,
        )
    elif method == "geoparquet":
        if geoparquet_file is None:
            raise ValueError("geoparquet_file required when method='geoparquet'")

        stac_items = _stac_items_from_geoparquet(
            geoparquet_file,
            collection,
            start_time,
            end_time,
            shape(intersects),
            filters=filters,
        )

    else:
        raise ValueError(
            f"Unknown method: {method!r}. Allowed methods are: {get_args(METHOD)}"
        )

    # Check size of stac items before proceeding to load
    if len(stac_items) == 0:
        raise ValueError("No STAC items found matching the search criteria")

    return stac_items


def load_sentinel_1_from_stac(
    item_collection: ItemCollection,
    band: str,
    band_unit: SAR_UNIT,
    groupby: str | None = "time",
    crs: CRS | str | int | None = None,
    resolution: float | int | None = None,
    bbox: BoundingBox | tuple[float, float, float, float] | None = None,
    geopolygon: GeoInterface | None = None,
    geobox: GeoBox | None = None,
    like: xr.DataArray | xr.Dataset | None = None,
) -> xr.DataArray:
    """Load a list of Sentinel-1 STAC items into an xarray.DataArray and
    add required metadata to prepare for use in the normalised product step

    Parameters
    ----------
    item_collection : ItemCollection
        A list of STAC items that can be passed to a loader.
    band : str
        The band to load (e.g. 'hh_gamma0').
    band_unit : SAR_UNIT
        The native units of the band, one of 'linear' or 'db'.
    groupby : str | None, optional
        How to group STAC items, by default 'time'. Alternative is 'sat:relative_orbit'.
    crs : CRS | str | int | None, optional
        Coordinate reference system to project to, by default None.
    resolution : float | int | None, optional
        Resolution to output at in the units of the output CRS, by default None.
    bbox : BoundingBox | tuple[float, float, float, float] | None, optional
        Bounding box to crop the loaded items to, by default None.
    geopolygon : GeoInterface | None, optional
        Geopolygon to crop the loaded items to, by default None.
    geobox : GeoBox | None, optional
        GeoBox to specify the CRS, resolution and geopolygon to crop to, by default None.
    like : xr.DataArray | xr.Dataset | None, optional
        xr.DataArray to specify the CRS, resolution and extent to crop to, by default None.

    Returns
    -------
    xr.DataArray
        An xr.DataArray containing Sentinel-1 data in decibels for use with the normalised product approach.

    Raises
    ------
    ValueError
        If no STAC items are supplied.
    """

    # Check size of stac items before proceeding to load
    if len(item_collection) == 0:
        raise ValueError("No STAC items provided")

    if isinstance(bbox, BoundingBox):
        bbox = bbox.bbox

    # Load the item into a dataset
    ds = load(
        item_collection,
        bands=band,
        groupby=groupby,
        chunks={},
        crs=crs,
        resolution=resolution,
        bbox=bbox,
        geopolygon=geopolygon,
        geobox=geobox,
        like=like,
    )

    da = ds[band].rename(band)
    da.attrs["unit"] = band_unit

    return _validate_and_standardise_sentinel1(da)
