import numpy as np

CLASS_LABELS = {
    0: "not-fast-ice",
    1: "mask",
    2: "fast-ice",
}

FAST_ICE_CLASS_VALUES = [2]

CLASS_COLORS = np.array([
    [128, 128, 128],   # class 0 (not-fast-ice) → grey
    [  0,   0,   0],   # class 1 (mask) → black
    [ 30, 144, 255],   # class 2 (fast-ice) → dodger blue
], dtype=np.uint8)