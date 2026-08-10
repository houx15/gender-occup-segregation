import textwrap

from scripts.common.config_loader import load_config


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_load_config_accepts_american_stories(tmp_path):
    cfg_path = _write(tmp_path, "am.yml", """
        language: "en"
        data_source: "american_stories"
        analysis_mode: "garg_weat"
        paths:
          base_dir: "/tmp/x"
          raw_data_dir: "/tmp/x/raw"
          corpora_dir: "/tmp/x/corpora"
          models_dir: "/tmp/x/models"
          results_dir: "/tmp/x/results"
          log_dir: "/tmp/x/logs"
          figures_dir: "/tmp/x/figures"
    """)
    cfg = load_config(cfg_path)
    assert cfg["data_source"] == "american_stories"
    # _CORPUS_DEFAULTS registration applied without KeyError:
    assert cfg["corpus"]["tokenizer"] == "nltk_en"


def test_load_config_accepts_dlnews(tmp_path):
    cfg_path = _write(tmp_path, "dl.yml", """
        language: "en"
        data_source: "dlnews"
        analysis_mode: "garg_weat"
        paths:
          base_dir: "/tmp/y"
          raw_data_dir: "/tmp/y/raw"
          corpora_dir: "/tmp/y/corpora"
          models_dir: "/tmp/y/models"
          results_dir: "/tmp/y/results"
          log_dir: "/tmp/y/logs"
          figures_dir: "/tmp/y/figures"
    """)
    cfg = load_config(cfg_path)
    assert cfg["data_source"] == "dlnews"
    assert cfg["corpus"]["stopwords"] == "en_default"
