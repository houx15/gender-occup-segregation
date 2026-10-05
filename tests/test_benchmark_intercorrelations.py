import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_intercorrelations import oriented_within_period_corr


def test_oriented_within_period_corr_flips_to_traditional_direction():
    rng = np.random.default_rng(0)
    rows = []
    for p in (2005, 2010):
        for i in range(30):
            trad = rng.normal()
            rows.append({"period": p, "state": f"s{i}",
                         "duncan": trad,                 # +1: higher = traditional
                         "female_emp_share": -trad,      # -1: higher = less traditional
                         "iat_sex_balanced": rng.normal()})
    m = oriented_within_period_corr(pd.DataFrame(rows),
                                    ["duncan", "female_emp_share", "iat_sex_balanced"])
    assert m.loc["duncan", "female_emp_share"] == pytest.approx(1.0)   # agree after orienting
    assert abs(m.loc["duncan", "iat_sex_balanced"]) < 0.4
    assert m.loc["duncan", "duncan"] == pytest.approx(1.0)
