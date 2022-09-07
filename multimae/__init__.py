from .criterion import MaskedCrossEntropyLoss, MaskedL1Loss, MaskedMSELoss
from .input_adapters import PatchedInputAdapter
from .multimae import MultiMAE, MultiViT
from .output_adapters import (LinearOutputAdapter,
                              SpatialOutputAdapter,
                              OutputAdapter)
