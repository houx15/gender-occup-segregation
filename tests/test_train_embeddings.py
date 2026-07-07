"""Tests for train_embeddings.CorpusIterator — compact count expansion.

The per_year_capped subsampled corpus is stored COMPACT: one line per unique
ngram per slice as ``ngram<TAB>count`` (count = Σ_year n_emit), instead of
``count`` repeated physical lines. This keeps disk at type-size (~presence)
while word2vec still sees ``count`` copies. CorpusIterator(expand_counts=True)
does the expansion at read time (Kozlowski et al. 2019 streaming style).
"""

from __future__ import annotations

from pathlib import Path

from scripts.train_embeddings import CorpusIterator


def _write_unit(corpora_dir: Path, unit: str, files: dict[str, list[str]]) -> None:
    d = corpora_dir / unit
    d.mkdir(parents=True, exist_ok=True)
    for name, lines in files.items():
        (d / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestExpandCounts:
    def test_expands_ngram_tab_count_into_repeated_sentences(self, tmp_path):
        _write_unit(tmp_path, "1940_1949", {
            "corpus_00000.txt": ["中国 经济\t3", "改革 政策\t2"],
        })
        it = CorpusIterator(str(tmp_path), "1940_1949", expand_counts=True)
        assert list(it) == [
            ["中国", "经济"], ["中国", "经济"], ["中国", "经济"],
            ["改革", "政策"], ["改革", "政策"],
        ]

    def test_line_without_count_column_yields_once(self, tmp_path):
        # Robustness: a bare ngram (no TAB) means one copy.
        _write_unit(tmp_path, "1940_1949", {
            "corpus_00000.txt": ["中国 经济"],
        })
        it = CorpusIterator(str(tmp_path), "1940_1949", expand_counts=True)
        assert list(it) == [["中国", "经济"]]

    def test_blank_lines_skipped(self, tmp_path):
        _write_unit(tmp_path, "1940_1949", {
            "corpus_00000.txt": ["中国 经济\t2", "", "改革 政策\t1"],
        })
        it = CorpusIterator(str(tmp_path), "1940_1949", expand_counts=True)
        assert list(it) == [["中国", "经济"], ["中国", "经济"], ["改革", "政策"]]

    def test_reiterable_for_multiple_epochs(self, tmp_path):
        # gensim re-iterates the corpus once per vocab pass + each epoch.
        _write_unit(tmp_path, "1940_1949", {
            "corpus_00000.txt": ["中国 经济\t2"],
        })
        it = CorpusIterator(str(tmp_path), "1940_1949", expand_counts=True)
        first = list(it)
        second = list(it)
        assert first == second == [["中国", "经济"], ["中国", "经济"]]


class TestPlainMode:
    def test_default_yields_each_line_once_as_tokens(self, tmp_path):
        # Non-subsampled corpora (presence, sentences) are unaffected: one
        # sentence per line, no count column, no expansion.
        _write_unit(tmp_path, "1940_1949", {
            "corpus_00000.txt": ["中国 经济 发展", "改革 政策"],
        })
        it = CorpusIterator(str(tmp_path), "1940_1949")  # expand_counts defaults False
        assert list(it) == [["中国", "经济", "发展"], ["改革", "政策"]]

    def test_reads_files_in_sorted_order(self, tmp_path):
        _write_unit(tmp_path, "1940_1949", {
            "corpus_00001.txt": ["second 文件"],
            "corpus_00000.txt": ["first 文件"],
        })
        it = CorpusIterator(str(tmp_path), "1940_1949")
        assert list(it) == [["first", "文件"], ["second", "文件"]]
