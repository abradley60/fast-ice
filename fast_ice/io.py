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


@runtime_checkable
class GeoInterface(Protocol):
    @property
    def __geo_interface__(self) -> dict: ...


def _validate_and_standardise_sentinel1(da: xr.DataArray) -> xr.DataArray:

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


def _build_cql2_filter(filters: dict | None, raw_filter: dict | None) -> dict | None:

    if raw_filter is not None and filters is not None:
        raise ValueError("Provide either filters or raw_filters, not both")

    if raw_filter is not None:
        return raw_filter

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
    raw_filter: dict | None = None,  # Custom filter dictionary using CQL2 standard
) -> ItemCollection:

    stac_client = Client.open(stac_endpoint)

    cql2_filter = _build_cql2_filter(filters, raw_filter)

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
    # TODO add logic for dealing with the geoparquet file being a URL
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

    return stac_geoparquet.to_item_collection(items_gdf)


def query_sentinel_1_stac(
    collection: str,
    start_time: datetime,
    end_time: datetime,
    intersects: GeoInterface,
    filters: dict | None = None,  # key-value pairs with the assumed operator "="
    raw_filter: dict | None = None,  # Custom filter dictionary using CQL2 standard
    *,
    method: METHOD = "api",
    api_endpoint: str | None = None,
    geoparquet_file: str | os.PathLike | None = None,
) -> ItemCollection:

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
            raw_filter=raw_filter,
        )
    elif method == "geoparquet":
        if geoparquet_file is None:
            raise ValueError("geoparquet_file required when method='geoparquet'")

        if raw_filter is not None:
            raise ValueError(
                "raw_filter (CQL2) is not supported for the geoparquet path — "
                "use filters= instead"
            )

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
