from ._sample_data_metadata import _SAMPLE_SCENES_METADATA, SAMPLE_SCENES
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


# Sample scene metadata class
@dataclass(frozen=True)
class SampleScene:
    path: Path
    timestamp: datetime
    band: str
    band_unit: str

    def __str__(self) -> str:
        return (
            f"SampleScene\n"
            f"  path:      {self.path}\n"
            f"  timestamp: {self.timestamp.isoformat()}\n"
            f"  band:      {self.band}\n"
            f"  unit:      {self.band_unit}"
        )

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
    ew_20191217_metadata = _SAMPLE_SCENES_METADATA["ew_20191217"]
    return SampleScene(
        path=Path(SAMPLE_SCENES.fetch(ew_20191217_metadata["filename"])),
        timestamp=ew_20191217_metadata["timestamp"],
        band=ew_20191217_metadata["band"],
        band_unit=ew_20191217_metadata["band_unit"],
    )


def fetch_ew_20191229():
    ew_20191217_metadata = _SAMPLE_SCENES_METADATA["ew_20191229"]
    return SampleScene(
        path=Path(SAMPLE_SCENES.fetch(ew_20191217_metadata["filename"])),
        timestamp=ew_20191217_metadata["timestamp"],
        band=ew_20191217_metadata["band"],
        band_unit=ew_20191217_metadata["band_unit"],
    )


def fetch_iw_20191219():
    ew_20191217_metadata = _SAMPLE_SCENES_METADATA["iw_20191219"]
    return SampleScene(
        path=Path(SAMPLE_SCENES.fetch(ew_20191217_metadata["filename"])),
        timestamp=ew_20191217_metadata["timestamp"],
        band=ew_20191217_metadata["band"],
        band_unit=ew_20191217_metadata["band_unit"],
    )
