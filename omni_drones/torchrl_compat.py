"""Compatibility shim for TorchRL API changes (0.3.x -> 0.11.x)."""
import torchrl

# Spec renames
from torchrl.data import Composite as CompositeSpec
from torchrl.data import Bounded as BoundedTensorSpec
from torchrl.data import Unbounded as UnboundedContinuousTensorSpec
from torchrl.data import TensorSpec
from torchrl.data import Categorical as DiscreteTensorSpec

__all__ = [
    "CompositeSpec",
    "BoundedTensorSpec", 
    "UnboundedContinuousTensorSpec",
    "TensorSpec",
    "DiscreteTensorSpec",
]
