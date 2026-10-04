from scripts.data_prep.download_project_implicit import select_year_files

NAMES = [
    "Gender-Career IAT.public.2005-2020-CSV.zip",
    "Gender-Career IAT.public.2010-CSV.zip",
    "Gender-Career_IAT.public.2010.zip",
    "Gender-Career_IAT.public.2011.zip",
    "Gender-Career IAT.public.2025.csv.zip",
    "Gender-Career IAT.public.2025.sav.zip",
    "Gender-Career_IAT_public_2010_codebook.xlsx",
]


def test_prefers_csv_and_falls_back_to_spss_zip():
    got = select_year_files(NAMES, [2010, 2011, 2025])
    assert got == {2010: "Gender-Career IAT.public.2010-CSV.zip",
                   2011: "Gender-Career_IAT.public.2011.zip",
                   2025: "Gender-Career IAT.public.2025.csv.zip"}


def test_missing_year_is_an_error():
    import pytest
    with pytest.raises(SystemExit, match="2012"):
        select_year_files(NAMES, [2012])
