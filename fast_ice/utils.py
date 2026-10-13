from typing import cast
import numpy as np
import xarray as xr


def convert_linear_to_db_xr(da: xr.DataArray, epsilon: float = 1e-10) -> xr.DataArray:

    # The function clips values less than epsilon to avoid introducing -Inf
    clipped = da.clip(min=epsilon)
    with xr.set_options(keep_attrs=True):
        db = cast(xr.DataArray, 10 * np.log10(clipped))

    return db

def assign_geodata_from_xr(arr, ref_da, dtype=None):
    """
    Wrap a plain np.ndarray in an xr.DataArray, assigning coordinates, CRS,
    and transform from a reference xr.DataArray of the same spatial shape.

    Parameters
    ----------
    arr    : np.ndarray — (y, x) or (y, x, band), same spatial shape as ref_da
    ref_da : xr.DataArray — reference array to copy dims/coords/CRS/transform from
             (must have a working .rio accessor)
    dtype  : output dtype to cast arr to. Defaults to arr's existing dtype if None.

    Returns
    -------
    xr.DataArray — arr wrapped with ref_da's spatial coords, CRS, and transform.
    """
    y_dim, x_dim = ref_da.dims[0], ref_da.dims[1]
    has_band = arr.ndim == 3

    # Sanity check shapes match before assigning coords
    ref_shape = (ref_da.sizes[y_dim], ref_da.sizes[x_dim])
    if arr.shape[:2] != ref_shape:
        raise ValueError(
            f"Shape mismatch: arr spatial shape {arr.shape[:2]} != "
            f"ref_da shape {ref_shape}"
        )

    if dtype is not None:
        arr = arr.astype(dtype)

    if has_band:
        dims = (y_dim, x_dim, "band")
        coords = {
            y_dim: ref_da.coords[y_dim],
            x_dim: ref_da.coords[x_dim],
        }
        # Reuse band labels from ref_da if it has a matching band count,
        # otherwise fall back to simple integer labels
        if "band" in ref_da.dims and ref_da.sizes["band"] == arr.shape[2]:
            coords["band"] = ref_da.coords["band"].values
        else:
            coords["band"] = np.arange(arr.shape[2])
    else:
        dims = (y_dim, x_dim)
        coords = {
            y_dim: ref_da.coords[y_dim],
            x_dim: ref_da.coords[x_dim],
        }

    da = xr.DataArray(arr, dims=dims, coords=coords, attrs=ref_da.attrs)
    da = da.rio.write_crs(ref_da.rio.crs)
    da = da.rio.write_transform(ref_da.rio.transform())

    return da