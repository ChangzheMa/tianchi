import numpy as np
from tools.make_sample_data import generate
from src import features, task2_capital, config


def _matrix(tmp_path):
    codes = generate(str(tmp_path), '20260618', n_per_type=5)
    return features.build_feature_matrix([(c, '20260618') for c in codes], str(tmp_path))


def test_capital_scores_shape(tmp_path):
    m = _matrix(tmp_path)
    g = task2_capital.behavior_groups(m)
    assert len(g) == len(m)
    scores = task2_capital.compute_capital_scores(m, g)
    assert scores.shape == (len(m), 3)
    assert np.isfinite(scores).all()


def test_intention_values(tmp_path):
    m = _matrix(tmp_path)
    it = task2_capital.judge_intention(m)
    assert len(it) == len(m)
    assert set(it).issubset(set(config.INTENTIONS))
