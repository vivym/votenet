from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.MASK_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedVoteNet"

# Path (a file path, or URL like https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.USE_COLOR = False
_C.INPUT.USE_HEIGHT = True
_C.INPUT.NUM_POINTS = 40000
_C.INPUT.RANDOM_X_FLIP = 0.5
_C.INPUT.RANDOM_Y_FLIP = 0.5
_C.INPUT.RANDOM_Z_ROTATION = (-5, 5)
_C.INPUT.RANDOM_SCALE = (0.85, 1.15)
_C.INPUT.AXIS_ALIGNED_BOX = False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ()


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.DATALOADER.REPEAT_THRESHOLD = 0.0


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_pointnet2_backbone"
# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks.
_C.MODEL.BACKBONE.FREEZE_AT = -1


# ---------------------------------------------------------------------------- #
# Vote generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.VOTE_GENERATOR = CN()
_C.MODEL.VOTE_GENERATOR.NAME = "StandardVoteGenerator"

_C.MODEL.VOTE_GENERATOR.VOTE_FACTOR = 1
_C.MODEL.VOTE_GENERATOR.SEED_FEATURE_DIM = 256
_C.MODEL.VOTE_GENERATOR.NUM_PROPOSALS = 256
_C.MODEL.VOTE_GENERATOR.SAMPLING_STRATEGY = "seed_fps"


# ---------------------------------------------------------------------------- #
# Proposal generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.PROPOSAL_GENERATOR = CN()
# Current proposal generators include "VotingRPN" and "PrecomputedProposals"
_C.MODEL.PROPOSAL_GENERATOR.NAME = "VotingRPN"


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY
_C.MODEL.RPN.CENTERNESS = False


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
# Number of foreground classes
_C.MODEL.ROI_HEADS.NUM_CLASSES = 18
_C.MODEL.ROI_HEADS.CENTERNESS = False
_C.MODEL.ROI_HEADS.SEED_FEATURE_DIM = 256
_C.MODEL.ROI_HEADS.GRID_SIZE = 3
_C.MODEL.ROI_HEADS.USE_EXP = True


# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.NAME = "StandardBoxHead"


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# See votenet/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
_C.SOLVER.IMS_PER_BATCH = 16

# The reference number of workers (GPUs) this config is meant to train with.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size if the actual number
# of workers during training is different from this reference.
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0


# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from votenet.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0
