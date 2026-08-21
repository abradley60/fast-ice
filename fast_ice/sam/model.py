import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from loguru import logger

from .constants import WEIGHTS_URLS, DEFAULT_WEIGHTS_DIR, DEFAULT_MASK_GENERATOR_SETTINGS


def load_mask_generator_settings(path):
    """Load `SamAutomaticMaskGenerator` keyword arguments from a JSON config file."""
    with open(path) as f:
        return json.load(f)


class SAMSegmenter:
    """
    Loads a Segment Anything Model (SAM) checkpoint and runs automatic mask
    generation on an RGB image, producing a non-overlapping segment ID map
    suitable for downstream classification (e.g. by the SVM in `fast_ice.svm`).

    Parameters
    ----------
    model_type : str
        One of "vit_h", "vit_l", "vit_b" — selects both the model
        architecture and which checkpoint to download/load.
    checkpoint : str or pathlib.Path, optional
        Path to a SAM .pth checkpoint. Downloaded automatically on `load()`
        if it doesn't exist. Defaults to
        `fast_ice/sam/weights/<checkpoint filename>`.
    device : str or torch.device, optional
        Device to run the model on. Defaults to CUDA if available, else CPU.
    mask_generator_settings : str or pathlib.Path, optional
        Path to a JSON config file of `SamAutomaticMaskGenerator` keyword
        arguments (e.g. `points_per_side`, `pred_iou_thresh`, ...). Defaults
        to `DEFAULT_MASK_GENERATOR_SETTINGS` (UTAS production settings); pass
        `FAST_MASK_GENERATOR_SETTINGS`  for a sparser, quicker config useful 
        for testing.
    mask_generator_kwargs : dict, optional
        Keyword arguments that override individual entries loaded from
        `mask_generator_settings`, without needing a separate config file.
    """

    def __init__(
        self,
        model_type="vit_h",
        checkpoint=None,
        device=None,
        mask_generator_settings=DEFAULT_MASK_GENERATOR_SETTINGS,
        mask_generator_kwargs=None,
    ):
        if model_type not in WEIGHTS_URLS:
            raise ValueError(f"model_type must be one of {list(WEIGHTS_URLS)}, got {model_type!r}")

        self.model_type = model_type
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint = Path(checkpoint) if checkpoint else DEFAULT_WEIGHTS_DIR / Path(WEIGHTS_URLS[model_type]).name

        self.mask_generator_settings = Path(mask_generator_settings)
        self.mask_generator_kwargs = load_mask_generator_settings(self.mask_generator_settings)
        if mask_generator_kwargs:
            self.mask_generator_kwargs.update(mask_generator_kwargs)

        self.sam = None
        self.mask_generator = None

    def download_weights(self):
        """Download the checkpoint for `self.model_type` if not already present."""
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if self.checkpoint.exists():
            logger.info(f"Weights already exist at {self.checkpoint}")
            return self.checkpoint

        url = WEIGHTS_URLS[self.model_type]
        logger.info(f"Downloading {self.model_type} weights from {url}...")
        urllib.request.urlretrieve(url, self.checkpoint)
        logger.info(f"Saved to {self.checkpoint}")
        return self.checkpoint

    def load(self):
        """Download weights (if needed), then build the SAM model and mask generator."""
        self.download_weights()

        logger.info(f"Loading SAM ({self.model_type}) onto {self.device}...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam = self.sam.to(device=self.device)
        logger.info(f"SAM device: {next(self.sam.parameters()).device}")

        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam, **self.mask_generator_kwargs)
        return self

    def generate(self, image):
        """
        Run SAM automatic mask generation on an RGB image.

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB image.

        Returns
        -------
        sam_result : list[dict]
            Raw output of `SamAutomaticMaskGenerator.generate` — one dict per
            detected mask (keys include "segmentation", "area", ...).
        """
        if self.mask_generator is None:
            self.load()

        sam_result = self.mask_generator.generate(image)
        logger.info(f"SAM complete — {len(sam_result)} segments found.")
        return sam_result

    @staticmethod
    def build_segment_id_map(sam_result):
        """
        Resolve overlapping SAM masks into a single non-overlapping segment ID
        map, using a "smallest area wins" rule — smaller segments are painted
        into unclaimed pixels first and cannot be overwritten by larger ones.

        Parameters
        ----------
        sam_result : list[dict]
            Output of `generate` / `SamAutomaticMaskGenerator.generate`.

        Returns
        -------
        segment_id_map : np.ndarray
            (H, W) uint16 array of segment IDs. 0 = unclaimed/unlabelled pixels.
        """
        masks = [mask["segmentation"] for mask in sam_result]
        h, w = masks[0].shape
        segment_id_map = np.zeros((h, w), dtype=np.uint16)

        mask_areas = [np.sum(mask) for mask in masks]
        sorted_indices = np.argsort(mask_areas)  # ascending: smallest first

        for segment_id, idx in enumerate(sorted_indices, start=1):
            # Only paint pixels not yet claimed by any (smaller) segment
            unclaimed = segment_id_map == 0
            segment_id_map[masks[idx] & unclaimed] = segment_id

        logger.info(
            f"Segment ID map built — {len(sorted_indices)} segments, "
            f"{np.sum(segment_id_map == 0)} unlabelled pixels."
        )
        return segment_id_map

    def predict(self, image):
        """
        Convenience method: run `generate` then `build_segment_id_map`.

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB image.

        Returns
        -------
        segment_id_map : np.ndarray
            (H, W) uint16 array of non-overlapping segment IDs.
        sam_result : list[dict]
            Raw SAM output for the same image.
        """
        sam_result = self.generate(image)
        segment_id_map = self.build_segment_id_map(sam_result)
        return segment_id_map, sam_result
