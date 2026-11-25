"""Box-prompted lesion segmentation (MRI-CORE / SAM) producing a mask manifest for ``mask_provider=local_files``.

Preparation stage only: ``train``/``validate`` never import this package. The model wrapper
(:mod:`.mri_core`) needs the optional ``.[segment]`` dependencies. :mod:`.geometry` and :mod:`.pipeline`
need neither torch nor the external repository. See docs/segmentation-mri-core.md.
"""
