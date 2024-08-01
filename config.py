from yacs.config import CfgNode as CN

_C = CN()

# Image Data Configuration
_C.DATASET = CN()
_C.DATASET.DIR = ""
_C.DATASET.CACHE_DIR = ""
_C.DATASET.IMAGE_EXT = ""
_C.DATASET.KEY_POINTS = 0
_C.DATASET.CACHED_IMAGE_SIZE = []
_C.DATASET.PIXEL_SIZE = []


# Augmentation Configuration
_C.DATASET.AUGMENTATION = CN()
_C.DATASET.AUGMENTATION.REVERSE_AXIS = False
_C.DATASET.AUGMENTATION.FLIP = False
_C.DATASET.AUGMENTATION.FLIP_PAIRS = []
_C.DATASET.AUGMENTATION.ROTATION_FACTOR = 5
_C.DATASET.AUGMENTATION.INTENSITY_FACTOR = 0.25
_C.DATASET.AUGMENTATION.SF = 0.2
_C.DATASET.AUGMENTATION.TRANSLATION_X = 25
_C.DATASET.AUGMENTATION.TRANSLATION_Y = 50
_C.DATASET.AUGMENTATION.ELASTIC_STRENGTH = 50
_C.DATASET.AUGMENTATION.ELASTIC_SMOOTHNESS = 10


# Training configuration
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.LR = 0.001
_C.TRAIN.EPOCHS = 10


# Model Configuration
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'Unet'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()