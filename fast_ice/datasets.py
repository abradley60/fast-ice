import pooch
from pathlib import Path

SAMPLE_SCENES = pooch.create(
    path=pooch.os_cache("fast_ice"),
    base_url="https://deant-data-public-dev.s3.ap-southeast-2.amazonaws.com/fast-ice/",
    registry={
        "ga_s1a_nrb_1-0-0_EW_20191217T112553_HH-gamma0_db_sample.tif": "sha256:43041201e40bf1cb248f1c8ff75530164c31684a5db384f18a7a485f0162c6f5",
        "ga_s1a_nrb_1-0-0_EW_20191229T112553_HH-gamma0_db_sample.tif": "sha256:1cd87e4fd99c2f03717c52659796757485666a625621b00716b582d2c2c70c8b",
        "ga_s1a_nrb_1-0-0_IW_20191219T124744_HH-gamma0_db_sample.tif": "sha256:12b8b0cf5c236dda8db5e9e84dc7dfa8693b84770895aa6785dd439ac1e2e0a1",
    },
    env="FASTICE_DATA_DIR",
)


def fetch_ew_20191217():
    """Return the file location of ga_s1a_nrb_1-0-0_EW_20191217T112553_HH-gamma0_db_sample GeoTIFF"""
    return Path(
        SAMPLE_SCENES.fetch(
            "ga_s1a_nrb_1-0-0_EW_20191217T112553_HH-gamma0_db_sample.tif"
        )
    )


def fetch_ew_20191229():
    """Return the file location of ga_s1a_nrb_1-0-0_EW_20191229T112553_HH-gamma0_db_sample.tif"""
    return Path(
        SAMPLE_SCENES.fetch(
            "ga_s1a_nrb_1-0-0_EW_20191229T112553_HH-gamma0_db_sample.tif"
        )
    )


def fetch_iw_20191219():
    """Return the file location of ga_s1a_nrb_1-0-0_IW_20191219T124744_HH-gamma0_db_sample.tif"""
    return Path(
        SAMPLE_SCENES.fetch(
            "ga_s1a_nrb_1-0-0_IW_20191219T124744_HH-gamma0_db_sample.tif"
        )
    )


def fetch_demo_scenes(names=None) -> list[Path]:
    """Download (or return cached) demo Sentinel-1 GeoTIFFs."""
    names = names or list(SAMPLE_SCENES.registry.keys())
    return [Path(SAMPLE_SCENES.fetch(n)) for n in names]
