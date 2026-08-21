from pathlib import Path

MODEL_TYPES = ["vit_h", "vit_l", "vit_b"]

WEIGHTS_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Mask generator settings, as JSON configs
CONFIG_DIR = Path(__file__).resolve().parent / "config"

# Dense production / published settings from UTAS team 
DEFAULT_MASK_GENERATOR_SETTINGS = CONFIG_DIR / "default.json"

# Sparse sampling grid, fewer/coarser masks — quick to run, useful for testing.
FAST_MASK_GENERATOR_SETTINGS = CONFIG_DIR / "fast.json"
