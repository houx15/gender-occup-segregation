from scripts.common.periods import window_label_map, year_windows


def test_no_width_means_one_unit_per_year():
    assert year_windows([2000, 2001], None) == [(2000, [2000]), (2001, [2001])]


def test_bins_keep_a_short_final_bin():
    assert year_windows(list(range(1995, 2007)), 5) == [
        (1995, [1995, 1996, 1997, 1998, 1999]),
        (2000, [2000, 2001, 2002, 2003, 2004]),
        (2005, [2005, 2006]),
    ]


def test_rolling_windows_overlap_and_are_full_only():
    w = year_windows(list(range(1995, 2025)), 10, step=5)
    assert [label for label, _ in w] == [1995, 2000, 2005, 2010, 2015]
    assert w[2] == (2005, list(range(2005, 2015)))
    assert w[-1][1] == list(range(2015, 2025))  # 2020-2029 would be truncated: dropped


def test_window_label_map_assigns_a_year_to_every_window_containing_it():
    m = window_label_map(list(range(2005, 2025)), 10, step=5)
    assert m[2012] == [2005, 2010]
    assert m[2005] == [2005]
    assert m[2024] == [2015]
