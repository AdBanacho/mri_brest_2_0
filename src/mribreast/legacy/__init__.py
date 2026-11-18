"""mribreast.legacy: compatibility with the reference repository (C:\\code\\mri_brest @ 127a2604).

* :mod:`~mribreast.legacy.checkpoints` - legacy checkpoint filename eras, ranking, fold inventory, weight loading
* :mod:`~mribreast.legacy.names` - legacy experiment/run names (build + parse)
* :mod:`~mribreast.legacy.importer` - historical CSV/JSON -> normalized table + retraining flags
* :mod:`~mribreast.legacy.exporter` - new evaluation -> legacy CSV/JSON layout + legacy summarizer port

Importing this package performs no I/O and imports no torch.
"""
