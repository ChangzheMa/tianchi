from src import config

def test_pattern_rules_wellformed():
    assert len(config.PATTERN_RULES) == config.N_PATTERNS == 10
    assert len(config.PATTERN_NAMES) == 10
    for name in config.PATTERN_NAMES:
        assert name in config.PATTERN_DESC
        conds = config.PATTERN_CONDITIONS[name]
        assert len(conds) >= 3
        for feat, direction in conds:
            assert direction in (-1, 0, 1)

def test_capital_config():
    assert config.CAPITAL_TYPES == ['游资', '量化', '散户']
    for t in config.CAPITAL_TYPES:
        assert len(config.SCORE_WEIGHTS[t]) >= 8
    assert len(config.BEHAVIOR_FEATURES) == 10
    assert config.PATTERN_RANGE == (0.06, 0.18)
    assert config.CAPITAL_RANGE == (0.22, 0.45)

def test_hq_col_map_has_levels():
    for i in range(1, 11):
        assert config.HQ_COL_MAP[f'申卖价{i}'] == f'ask_price_{i}'
        assert config.HQ_COL_MAP[f'申买量{i}'] == f'bid_vol_{i}'
