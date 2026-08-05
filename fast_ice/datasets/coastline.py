import pooch
from pathlib import Path

def fetch_add_coastline_shapefile(output_dir):
    """
    Download and unzip the SCAR ADD medium-res coastline polygon shapefile
    from the BAS RAMADDA repository, returning the path to the .shp file.

    Parameters
    ----------
    output_dir : str or pathlib.Path — directory to download/unzip into

    Returns
    -------
    pathlib.Path to the extracted .shp file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://ramadda.data.bas.ac.uk/repository/entry/get/"
        "add_coastline_medium_res_polygon_v7_12.shp.zip?entryid=synth%3Ac9c6d671-ad19-4a78-96b2-7d47e9bac46a"
        "%3AL2FkZF9jb2FzdGxpbmVfbWVkaXVtX3Jlc19wb2x5Z29uX3Y3XzEyLnNocC56aXA%3D"
    )

    fnames = pooch.retrieve(
        url=url,
        known_hash="b0172d698436f871f49339cea5c7155e3d98648360521cc4bbc44a026c0edc38",
        fname="add_coastline_medium_res_polygon_v7_12.shp.zip",
        path=output_dir,
        processor=pooch.Unzip(extract_dir="add_coastline_medium_res_polygon_v7_12"),
    )

    shp_path = next(
        Path(f) for f in fnames
        if f.endswith("add_coastline_medium_res_polygon_v7_12.shp")
    )
    return shp_path