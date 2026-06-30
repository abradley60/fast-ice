# Module for loading from various sources and returning xarrays
import rioxarray
import geopandas as gpd
import stac_geoparquet
from odc.stac import load
import os


def _load_from_file(filename: str | os.PathLike):
    return rioxarray.open_rasterio(filename).squeeze()


# def _load_from_stac_api()


def _filter_geoparquet(
    gdf: gpd.GeoDataFrame,
    collection,
    start_time,
    end_time,
    aoi_bbox,
    filters: dict,
):

    # Collection
    gdf = gdf[gdf["collection"] == collection]
    # Area
    gdf = gdf[gdf.intersects(aoi_bbox.polygon.geom)]
    # Date
    gdf = gdf[(gdf.datetime >= start_time) & (gdf.datetime <= end_time)]
    # Other filters
    for key, value in filters.items():
        gdf = gdf[gdf[key] == value]

    return gdf


def _load_from_stac_geoparquet(
    geoparquet_file: str | os.PathLike,
    collection,
    start_time,
    end_time,
    aoi_bbox,
    resolution,
    crs,
    filters: dict,
):

    # Assume local geoparquet for now
    geoparquet_gdf = gpd.read_parquet(geoparquet_file)

    filtered_gdf = _filter_geoparquet(
        geoparquet_gdf, collection, start_time, end_time, aoi_bbox, filters
    )

    # convert to stac collection
    stac_items = stac_geoparquet.to_item_collection(filtered_gdf)

    ds = load(
        stac_items, chunks={}, resolution=resolution, crs=crs, bbox=aoi_bbox.bbox
    ).squeeze()

    return ds
