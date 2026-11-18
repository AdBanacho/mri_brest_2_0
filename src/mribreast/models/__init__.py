"""mribreast.models: 3D/2D networks, the Lightning image classifier and the image branch.

Heavy modules (Lightning/MONAI) are imported lazily by the submodules; importing
this package performs no I/O.
"""

from mribreast.models.networks import MRI_MODELS, Simple3DFCN, SimpleFCN, make_network

__all__ = ["MRI_MODELS", "Simple3DFCN", "SimpleFCN", "make_network"]
