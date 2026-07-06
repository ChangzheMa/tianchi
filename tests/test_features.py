from tools.make_sample_data import generate
from src import features
from src import config

def test_extract_features_keys(tmp_path):
    codes = generate(str(tmp_path), '20260618', n_per_type=1)
    f = features.extract_features(codes[0], '20260618', str(tmp_path))
    assert f['stock_code'] == codes[0] and f['transaction_date'] == '20260618'
    for k in ['oss_mega_amount_pct', 'trd_total_count', 'rs_interval_cv', 'cb_cancel_ratio',
              'ap_active_buy_pct', 'pi_time_concentration', 'pd_day_return', 'book_imbalance',
              'obp_rel_spread_mean', 'fd_herfindahl']:
        assert k in f, f'缺特征 {k}'

def test_pattern_and_score_features_present(tmp_path):
    codes = generate(str(tmp_path), '20260618', n_per_type=1)
    f = features.extract_features(codes[0], '20260618', str(tmp_path))
    referenced = {c for conds in config.PATTERN_CONDITIONS.values() for c, _ in conds}
    referenced |= {c for ws in config.SCORE_WEIGHTS.values() for c, _ in ws}
    referenced |= set(config.BEHAVIOR_FEATURES)
    missing = referenced - set(f.keys())
    assert not missing, f'配置引用但未产出的特征: {missing}'

def test_build_matrix(tmp_path):
    codes = generate(str(tmp_path), '20260618', n_per_type=2)
    pairs = [(c, '20260618') for c in codes]
    m = features.build_feature_matrix(pairs, str(tmp_path))
    assert len(m) == 6
    assert not m.isnull().any().any()
