import re
import numpy as np
import joblib
from pathlib import Path
from sklearn import svm

from loguru import logger

def train(train_data_dir, outpath):
    """
    Train an SVM classifier on manually labelled SAM segments and save the
    trained model to disk.

    Iterates through each subfolder of `train_data_dir`, where each subfolder
    corresponds to a labelled scene (e.g. "prydz_20210129_20210210") and
    contains:
        - HH_rgb_image_<dates>.npy    : (H, W, 3) NormProd RGB composite
        - label_map_<dates>.npy       : (H, W) SAM segment ID map
        - labelled_array_<dates>.npy  : per-segment manual class labels
                                         (NaN = unlabelled)

    For each labelled segment, the mean RGB value across its pixels is used
    as a feature vector. An SVM is trained on these features and saved to
    `outpath`.

    Parameters
    ----------
    train_data_dir : str or pathlib.Path
        Directory containing one subfolder per labelled training scene.
    outpath : str or pathlib.Path
        File path to save the trained model weights to (e.g. "svm_weights.joblib").

    Returns
    -------
    clf : sklearn.svm.SVC
        The trained SVM classifier.
    """
    train_data_dir = Path(train_data_dir)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    X_train = []
    y_train = []

    training_image_list = sorted(
        d.name for d in train_data_dir.iterdir() if d.is_dir()
    )

    for scene in training_image_list:
        logger.info(f"Loading training scene: {scene}")

        scene_dir = train_data_dir / scene

        # Extract date string from folder name
        # e.g. prydz_20210129_20210210 → 20210129_20210210
        dates = re.findall(r'\d{8}', scene)
        if len(dates) < 2:
            logger.info(f"Could not extract dates from {scene}, skipping.")
            continue
        scene_date = f"{dates[0]}_{dates[1]}"

        rgb_path = scene_dir / f"HH_rgb_image_{scene_date}.npy"
        label_map_path = scene_dir / f"label_map_{scene_date}.npy"
        labelled_array_path = scene_dir / f"labelled_array_{scene_date}.npy"

        if not (rgb_path.exists() and label_map_path.exists() and labelled_array_path.exists()):
            logger.warning(f"Missing expected files in {scene}, skipping.")
            continue

        # Load RGB, segment ID map, and manual segment labels
        scene_rgb_landmask_eroded = np.load(rgb_path)
        segment_id_map = np.load(label_map_path)
        segment_labels = np.load(labelled_array_path, allow_pickle=True)

        # Track labels for this scene specifically
        scene_labels = []

        # For each segment, extract mean RGB as training feature vector
        for seg_id in np.unique(segment_id_map):
            seg_label = segment_labels[seg_id]

            # Skip unlabelled segments (NaN = not manually annotated)
            if np.isnan(seg_label):
                continue

            # Mean RGB across all pixels in this segment → 1x3 feature vector
            seg_pixels = scene_rgb_landmask_eroded[segment_id_map == seg_id]
            X_train.append(seg_pixels.mean(axis=0).tolist())
            y_train.append(int(seg_label))
            scene_labels.append(int(seg_label))

        logger.info(f"Unique classes in {scene}: {sorted(set(scene_labels))} "
              f"({len(set(scene_labels))} classes, {len(scene_labels)} labelled segments)")
        logger.info(f"Running totals — X_train: {len(X_train)}, y_train: {len(y_train)}")

    X_train = np.array(X_train)  # shape: (n_segments, 3)
    y_train = np.array(y_train)  # shape: (n_segments,)

    logger.info(f"\nTraining data built — {len(X_train)} segments across {len(training_image_list)} scenes.")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Train SVM
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    logger.info(f"SVM trained on {len(X_train)} segments.")
    logger.info(f"Classes: {clf.classes_}")

    # Save weights
    joblib.dump(clf, outpath)
    logger.info(f"\nModel saved to {outpath}")

    return clf