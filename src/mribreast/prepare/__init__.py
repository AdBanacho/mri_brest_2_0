"""mribreast.prepare: explicit data-preparation stage (``pip install .[prepare]``).

* :mod:`~mribreast.prepare.nbia` - NBIA/TCIA REST download (injected HTTP client)
* :mod:`~mribreast.prepare.dicom_to_nifti` - DICOM -> ``<SeriesInstanceUID>.nii.gz`` after UID + geometry checks
* :mod:`~mribreast.prepare.mama_mia` - MAMA-MIA Synapse resource + mapping onto Duke series
* ``python -m mribreast.prepare <command>`` - command line (:mod:`~mribreast.prepare.cli`)

No network access and no optional-dependency import at package import time.
"""
