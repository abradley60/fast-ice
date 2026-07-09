import pooch
from pathlib import Path

EW_STAC_GEOPARQUET_FILENAME = "ga_s1_nrb_ew_hh_hv_1_v2.parquet"

EW_STAC_GEOPARQUET = pooch.create(
    path=pooch.os_cache("fast_ice"),
    base_url="https://data.dev.dea.ga.gov.au/experimental/baseline/pyrosar_gamma/",
    registry={
        EW_STAC_GEOPARQUET_FILENAME: "sha256:62eab1280286cb2c4a15f51237d4710620efa03b9889f72583acc4bf96127515"
    },
    env="FASTICE_STAC_GEOPARQUET_DIR",
)


def download_ew_stac_geoparquet() -> Path:

    return Path(EW_STAC_GEOPARQUET.fetch(EW_STAC_GEOPARQUET_FILENAME, progressbar=True))
