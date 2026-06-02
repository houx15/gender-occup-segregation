"""Per-source raw-data walkers for the dataset summary reporter.

Each walker exports a ``walk(raw_data_dir, units, config, logger)`` function
returning ``Dict[unit_name, RawVolumeEntry]``. The WALKERS registry maps
config ``data_source`` strings to walkers; the reporter dispatches on it.
"""

from scripts.data_prep.raw_volume.rmrb import walk as _walk_rmrb
from scripts.data_prep.raw_volume.provincial_newspaper import walk as _walk_newspaper
from scripts.data_prep.raw_volume.weibo import walk as _walk_weibo
from scripts.data_prep.raw_volume.ngram_zh import walk as _walk_ngram_zh
from scripts.data_prep.raw_volume.ngram_en import walk as _walk_ngram_en
from scripts.data_prep.raw_volume.coha import walk as _walk_coha

WALKERS = {
    "renminribao": _walk_rmrb,
    "newspaper": _walk_newspaper,
    "weibo": _walk_weibo,
    # ngram is dispatched by (language, data_source) — see resolve_walker()
    # in scripts.common.dataset_stats. The registry uses synthetic keys.
    "ngram_zh": _walk_ngram_zh,
    "ngram_en": _walk_ngram_en,
    "coha": _walk_coha,
}
