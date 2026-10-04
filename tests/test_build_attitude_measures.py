import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.data_prep.build_attitude_measures import (
    attitude_measures, harmonize, read_year_file,
)


def test_harmonize_sex_codings_and_us_states():
    raw = pd.DataFrame({
        "year": [2010, 2010, 2017, 2024, 2024],
        "STATE": ["CA", "ZZ", "TX ", "NY", "NY"],
        "sex": ["f", "m", None, None, None],
        "birthsex": [None, None, 1, None, None],
        "genderIdentity_0002": [None, None, None, 2, 3],
        "D_biep.Male_Career_all": [0.4, 0.1, -0.2, 0.3, 0.5],
        "assocareer": [6, 5, 4, 7, 4],
        "assofamily": [2, 3, 4, 1, 4],
    })
    h = harmonize(raw, file_year=2010)
    assert list(h["state"]) == ["CA", "TX", "NY", "NY"]          # ZZ dropped
    assert h["female"].iloc[:3].tolist() == [1.0, 0.0, 1.0]   # sex f, birthsex male, woman
    assert np.isnan(h["female"].iloc[3])                      # non-binary: no binary sex
    assert list(h["explicit"]) == [4, 0, 6, 0]


def test_read_csv_zip_member(tmp_path):
    z = tmp_path / "Gender-Career IAT.public.2010-CSV.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("__MACOSX/._x.csv", "junk")
        f.writestr("Gender-Career IAT.public.2010.csv",
                   "year,STATE,sex,D_biep.Male_Career_all,assocareer,assofamily,extra\n"
                   "2010,CA,f,0.4,6,2,1\n")
    df = read_year_file(z, tmp_path)
    assert list(df.columns) == ["year", "STATE", "sex", "D_biep.Male_Career_all",
                                "assocareer", "assofamily"]


def test_attitude_measures_raw_and_sex_balanced():
    h = pd.DataFrame({
        "year": [2005] * 4 + [2012],
        "state": ["CA"] * 5,
        "female": [1, 1, 1, 0, 1],
        "iat": [0.4, 0.4, 0.4, 0.0, 9.0],
        "explicit": [2, 2, 2, 0, 9],
    })
    m = attitude_measures(h, period_start=2005, width=5).set_index(["state", "period"])
    r = m.loc[("CA", 2005)]
    assert r["iat_mean"] == pytest.approx(0.3)
    assert r["iat_sex_balanced"] == pytest.approx(0.2)      # (0.4 + 0.0) / 2
    assert r["explicit_sex_balanced"] == pytest.approx(1.0)
    assert r["n"] == 4
