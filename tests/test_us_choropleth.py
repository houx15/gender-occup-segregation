import pandas as pd
from scripts import visualize as v


def test_state_year_parse():
    assert v._state_year_parse("new_york_1940") == ("New York", 1940)
    assert v._state_year_parse("california_1996") == ("California", 1996)
    assert v._state_year_parse("district_of_columbia_2000") == ("District of Columbia", 2000)
    assert v._state_year_parse("1990s") is None
    assert v._state_year_parse("not_a_unit") is None


def test_match_state_in_shapefile_joins_on_name():
    gpd = __import__("importlib").import_module("geopandas") if _has_gpd() else None
    if gpd is None:
        import pytest; pytest.skip("geopandas not installed")
    from shapely.geometry import Point
    states = gpd.GeoDataFrame(
        {"NAME": ["California", "Nevada"], "geometry": [Point(0, 0), Point(1, 1)]})
    dim = pd.DataFrame({"state": ["California"], "oriented_rnd": [0.12]})
    merged = v._match_state_in_shapefile(dim, states)
    row = merged[merged["NAME"] == "California"].iloc[0]
    assert abs(row["oriented_rnd"] - 0.12) < 1e-9


def _has_gpd():
    import importlib.util
    return importlib.util.find_spec("geopandas") is not None
