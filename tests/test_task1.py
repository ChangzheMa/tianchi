import numpy as np
from tools.make_sample_data import generate
from src import features, task1_pattern, config


def _matrix(tmp_path):
    codes = generate(str(tmp_path), '20260618', n_per_type=5)  # 15 只
    return features.build_feature_matrix([(c, '20260618') for c in codes], str(tmp_path))


def test_normalize_shape(tmp_path):
    m = _matrix(tmp_path)
    arr, cols = task1_pattern.normalize_robust(m)
    assert arr.shape[0] == len(m) and len(cols) == arr.shape[1]
    assert np.isfinite(arr).all()


def test_clustering_and_mapping(tmp_path):
    m = _matrix(tmp_path)
    arr, _ = task1_pattern.normalize_robust(m)
    labels = task1_pattern.run_clustering(arr, k=min(config.N_PATTERNS, len(m)))
    assert len(labels) == len(m)
    scores = task1_pattern.compute_pattern_scores(m)
    assert scores.shape == (len(m), config.N_PATTERNS)
    pat = task1_pattern.map_clusters_to_patterns(labels, scores)
    assert len(pat) == len(m)
    assert set(pat).issubset(set(config.PATTERN_NAMES))
