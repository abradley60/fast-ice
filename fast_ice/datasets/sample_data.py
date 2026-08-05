import pooch
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from fast_ice.io import SAR_UNIT

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


# Sample scene metadata class
@dataclass(frozen=True)
class SampleScene:
    path: Path
    timestamp: datetime
    band: str
    band_unit: SAR_UNIT

    # Add to allow nice rendering in notebook
    def _repr_html_(self) -> str:
        return (
            f"<b>SampleScene</b><table>"
            f"<tr><td>path</td><td>{self.path}</td></tr>"
            f"<tr><td>timestamp</td><td>{self.timestamp.isoformat()}</td></tr>"
            f"<tr><td>band</td><td>{self.band}</td></tr>"
            f"<tr><td>unit</td><td>{self.band_unit}</td></tr>"
            f"</table>"
        )


def fetch_ew_20191217():
    """Download s1a_EW_20191217_hh_gamma0_db_sample.tif stored on AWS

    Returns
    -------
    SampleScene
        A dataclass with the local path to the downloaded file, along with
        the timestamp, band, and band unit for the file.
    """
    return SampleScene(
        path=Path(SAMPLE_SCENES.fetch("s1a_EW_20191217_hh_gamma0_db_sample.tif")),
        timestamp=datetime(2019, 12, 17, 11, 25, 53),
        band="hh_gamma0",
        band_unit="db",
    )


def fetch_ew_20191229():
    """Download s1a_EW_20191229_hh_gamma0_db_sample.tif stored on AWS

    Returns
    -------
    SampleScene
        A dataclass with the local path to the downloaded file, along with
        the timestamp, band, and band unit for the file.
    """
    return SampleScene(
        path=Path(SAMPLE_SCENES.fetch("s1a_EW_20191229_hh_gamma0_db_sample.tif")),
        timestamp=datetime(2019, 12, 29, 11, 25, 53),
        band="hh_gamma0",
        band_unit="db",
    )


def fetch_iw_20191219():
    """Download s1a_IW_20191219_hh-gamma0_db_sample.tif stored on AWS

    Returns
    -------
    SampleScene
        A dataclass with the local path to the downloaded file, along with
        the timestamp, band, and band unit for the file.
    """
    return SampleScene(
        path=Path(SAMPLE_SCENES.fetch("s1a_IW_20191219_hh-gamma0_db_sample.tif")),
        timestamp=datetime(2019, 12, 19, 12, 47, 44),
        band="hh_gamma0",
        band_unit="db",
    )
