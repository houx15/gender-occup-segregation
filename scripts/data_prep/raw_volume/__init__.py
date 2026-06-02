"""Per-source raw-data walkers for the dataset summary reporter.

Each walker exports a ``walk(raw_data_dir, units, config, logger)`` function
returning ``Dict[unit_name, RawVolumeEntry]``. The WALKERS registry maps
config ``data_source`` strings to walkers; the reporter dispatches on it.
"""

from scripts.data_prep.raw_volume.rmrb import walk as _walk_rmrb

WALKERS = {
    "renminribao": _walk_rmrb,
}
