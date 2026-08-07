import matplotlib.pyplot as plt
import supervision as sv

from loguru import logger


def annotate_masks(image, sam_result):
    """
    Overlay SAM's raw (possibly overlapping) masks onto an image for visual
    inspection, using `supervision`'s per-instance colour lookup.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) uint8 RGB image the masks were generated from.
    sam_result : list[dict]
        Output of `SAMSegmenter.generate` / `SamAutomaticMaskGenerator.generate`.

    Returns
    -------
    annotated_image : np.ndarray
        (H, W, 3) uint8 image with masks drawn on top of `image`.
    """
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(
        scene=image.copy(),
        detections=detections,
    )
    return annotated_image


def plot_segmentation(image, annotated_image, title=None, save_path=None):
    """
    Plot the source image alongside its SAM segmentation overlay.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) source RGB image.
    annotated_image : np.ndarray
        (H, W, 3) image with SAM masks drawn on top, e.g. as returned by
        `annotate_masks`.
    title : str, optional
        Overall figure title (e.g. f"mode {mode}, dates: {date_str}").
    save_path : str or pathlib.Path, optional
        If given, saves the figure to this path.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title("a) Source image")
    axes[0].axis("off")

    axes[1].imshow(annotated_image)
    axes[1].set_title("b) SAM segmentation")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved: {save_path}")

    plt.show()
    plt.close(fig)

    return fig, axes


def plot_segment_id_map(segment_id_map, title=None, save_path=None):
    """
    Plot a non-overlapping segment ID map, e.g. as returned by
    `SAMSegmenter.build_segment_id_map`.

    Parameters
    ----------
    segment_id_map : np.ndarray
        (H, W) array of segment IDs. 0 = unclaimed/unlabelled pixels.
    title : str, optional
        Plot title. Defaults to a generic description.
    save_path : str or pathlib.Path, optional
        If given, saves the figure to this path.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(segment_id_map, cmap="gist_ncar")
    ax.set_title(title or "Segment ID map — smallest area wins")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Segment ID")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved: {save_path}")

    plt.show()
    plt.close(fig)

    return fig, ax
