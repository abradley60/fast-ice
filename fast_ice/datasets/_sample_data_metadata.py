import pooch
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Pooch Registry
SAMPLE_SCENES = pooch.create(
    path=pooch.os_cache("fast_ice"),
    base_url="https://deant-data-public-dev.s3.ap-southeast-2.amazonaws.com/fast-ice/",
    registry={
        "s1a_EW_20191217_hh_gamma0_db_sample.tif": "sha256:43041201e40bf1cb248f1c8ff75530164c31684a5db384f18a7a485f0162c6f5",
        "s1a_EW_20191229_hh_gamma0_db_sample.tif": "sha256:1cd87e4fd99c2f03717c52659796757485666a625621b00716b582d2c2c70c8b",
        "s1a_IW_20191219_hh-gamma0_db_sample.tif": "sha256:12b8b0cf5c236dda8db5e9e84dc7dfa8693b84770895aa6785dd439ac1e2e0a1",
    },
    env="FASTICE_DATA_DIR",
)

_SAMPLE_SCENES_METADATA: dict[str, dict[str, Any]] = {
    # Scene metadata for s1a_EW_20191217_hh_gamma0_db_sample.tif
    "ew_20191217": dict(
        filename="s1a_IW_20191219_hh-gamma0_db_sample.tif",
        timestamp=datetime(2019, 12, 17, 11, 25, 53),
        band="hh_gamma0",
        band_unit="db",
    ),
    # Scene metadata for s1a_EW_20191229_hh_gamma0_db_sample.tif
    "ew_20191229": dict(
        filename="s1a_EW_20191229_hh_gamma0_db_sample.tif",
        timestamp=datetime(2019, 12, 29, 11, 25, 53),
        band="hh_gamma0",
        band_unit="db",
    ),
    # Scene metadata for s1a_IW_20191219_hh-gamma0_db_sample.tif
    "iw_20191219": dict(
        filename="s1a_IW_20191219_hh-gamma0_db_sample.tif",
        timestamp=datetime(2019, 12, 19, 12, 47, 44),
        band="hh_gamma0",
        band_unit="db",
    ),
}
