import numpy as np
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from .constants import CLASS_LABELS, CLASS_COLORS

from loguru import logger

def predict_on_image_and_segments(rgb, segment_id_map, model):
    """
    Predict a class for each segment in a SAM segment ID map, using the
    mean RGB value of each segment as the feature vector for a trained SVM.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) RGB image the segments were generated from (e.g.
        `results["rgb_resampled_landmasked"]`).
    segment_id_map : np.ndarray
        (H, W) array of segment IDs, e.g. as built by the "smallest area
        wins" logic — 0 = unclaimed/unlabelled pixels.
    model : str, pathlib.Path, or sklearn estimator
        Either a path to trained SVM weights (as saved by `train`, via
        joblib), or an already-loaded classifier object.

    Returns
    -------
    pred_map : np.ndarray
        (H, W) uint8 array — each pixel painted with its segment's
        predicted class. Unclaimed pixels (segment_id == 0) are left as 0.
    seg_predictions : dict[int, int]
        Mapping of segment_id -> predicted class.
    """
    # Accept either a path to weights, or an already-loaded model
    if isinstance(model, (str, Path)):
        clf = joblib.load(model)
    else:
        clf = model

    # Only predict on real segments, skip unclaimed background (id 0)
    seg_ids = np.unique(segment_id_map)
    seg_ids = seg_ids[seg_ids != 0]

    # Mean RGB per segment -> feature vector
    X = np.array([
        rgb[segment_id_map == seg_id].mean(axis=0)
        for seg_id in seg_ids
    ])

    y_pred = clf.predict(X)
    seg_predictions = dict(zip(seg_ids.tolist(), y_pred.tolist()))

    # Paint each segment with its predicted class
    pred_map = np.zeros_like(segment_id_map, dtype=np.uint8)
    for seg_id, pred_class in seg_predictions.items():
        pred_map[segment_id_map == seg_id] = pred_class

    logger.info(f"Predicted {len(seg_ids)} segments.")
    logger.info(f"Prediction distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}")

    return pred_map, seg_predictions

def plot_predictions(
    rgb,
    pred_map,
    class_colors=None,
    class_labels=None,
    title=None,
    save_path=None,
):
    """
    Plot the source RGB image alongside its predicted class map.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) source RGB image, e.g. `results["rgb_resampled_landmasked"]`.
    pred_map : np.ndarray
        (H, W) array of predicted class indices per pixel, e.g. as returned
        by `predict_on_image_and_segments`. Values are used to index into
        `class_colors`.
    class_colors : np.ndarray, optional
        (n_classes, 3) uint8 array of RGB colours, indexed by class value.
        Defaults to `DEFAULT_CLASS_COLORS`.
    class_labels : dict[int, str], optional
        Mapping of class index -> display label, used to build the legend.
        Defaults to `DEFAULT_CLASS_LABELS`.
    title : str, optional
        Overall figure title (e.g. f"mode {mode}, dates: {date_str}").
    save_path : str or pathlib.Path, optional
        If given, saves the figure to this path.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    class_colors = CLASS_COLORS if class_colors is None else class_colors
    class_labels = CLASS_LABELS if class_labels is None else class_labels

    rgb_pred_map = class_colors[pred_map]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(rgb)
    axes[0].set_title("a) Source image")
    axes[0].axis("off")

    axes[1].imshow(rgb_pred_map)
    axes[1].set_title("b) Predicted classes")
    axes[1].axis("off")

    # Build legend from classes actually present in pred_map
    present_classes = np.unique(pred_map)
    legend_handles = [
        mpatches.Patch(
            color=class_colors[c] / 255,
            label=class_labels.get(int(c), f"class {c}"),
        )
        for c in present_classes
        if c < len(class_colors)
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_handles),
        frameon=False,
    )

    if title:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    plt.show()
    plt.close(fig)

    return fig, axes