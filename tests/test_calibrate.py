import numpy as np
from src import calibrate

def test_calibrate_rebalances():
    rng = np.random.default_rng(0)
    labels = np.array(['A'] * 100, dtype=object)
    scores = rng.random((100, 3))
    names = ['A', 'B', 'C']
    out = calibrate.calibrate_distribution(labels, scores, names, 0.22, 0.45, rounds=8)
    counts = {c: (out == c).sum() for c in names}
    assert all(22 <= counts[c] <= 46 for c in names), counts
    assert len(out) == 100

def test_calibrate_preserves_when_balanced():
    labels = np.array((['A'] * 34 + ['B'] * 33 + ['C'] * 33), dtype=object)
    scores = np.zeros((100, 3))
    for i, c in enumerate(labels):
        scores[i, ['A', 'B', 'C'].index(c)] = 1.0
    out = calibrate.calibrate_distribution(labels, scores, ['A', 'B', 'C'], 0.22, 0.45)
    assert (out == labels).mean() > 0.9
