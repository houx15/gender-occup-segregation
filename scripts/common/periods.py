"""Time windows for state-period units: yearly, non-overlapping bins, or rolling.

Shared by the US corpus builder and the ACS share builder so that a text unit
and its survey counterpart always cover the same years. Windows are labelled
by their start year (``ohio_2005``).

- width None or 1: one unit per year.
- width N, step None or N: consecutive N-year bins from the first year; a
  short final bin is kept.
- width N, step S < N: overlapping rolling windows (e.g. 10 years every 5);
  only full windows are kept, so every rolling unit spans N years.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def year_windows(years: List[int], width: Optional[int],
                 step: Optional[int] = None) -> List[Tuple[int, List[int]]]:
    years = sorted(years)
    if not width or width == 1:
        return [(y, [y]) for y in years]
    step = step or width
    first, last = years[0], years[-1]
    out = []
    start = first
    while start <= last:
        span = [y for y in years if start <= y < start + width]
        full = start + width - 1 <= last
        if span and (full or step >= width):
            out.append((start, span))
        start += step
    return out


def window_label_map(years: List[int], width: Optional[int],
                     step: Optional[int] = None) -> Dict[int, List[int]]:
    """year -> labels of every window that contains it."""
    m: Dict[int, List[int]] = {}
    for label, span in year_windows(years, width, step):
        for y in span:
            m.setdefault(y, []).append(label)
    return m
