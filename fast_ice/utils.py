from typing import cast
import numpy as np
import xarray as xr


def convert_linear_to_db_xr(da: xr.DataArray, epsilon: float = 1e-10) -> xr.DataArray:

    # The function clips values less than epsilon to avoid introducing -Inf
    clipped = da.clip(min=epsilon)
    with xr.set_options(keep_attrs=True):
        db = cast(xr.DataArray, 10 * np.log10(clipped))

    return db
