"""全局常量与规则表（无逻辑）。"""

RANDOM_SEED = 42

# ── 行情.csv 中文列名 → 英文 ──
HQ_COL_MAP = {
    '万得代码': 'code', '交易所代码': 'exch', '自然日': 'date', '时间': 'time',
    '成交价': 'price', '成交量': 'volume', '成交额': 'amount', '成交笔数': 'trade_count',
    'BS标志': 'bs_flag', '当日累计成交量': 'cum_volume', '当日成交额': 'cum_amount',
    '最高价': 'high', '最低价': 'low', '开盘价': 'open', '前收盘': 'prev_close',
    '加权平均叫卖价': 'weighted_ask', '加权平均叫买价': 'weighted_bid',
    '叫卖总量': 'total_ask_vol', '叫买总量': 'total_bid_vol',
}
for _i in range(1, 11):
    HQ_COL_MAP[f'申卖价{_i}'] = f'ask_price_{_i}'
    HQ_COL_MAP[f'申卖量{_i}'] = f'ask_vol_{_i}'
    HQ_COL_MAP[f'申买价{_i}'] = f'bid_price_{_i}'
    HQ_COL_MAP[f'申买量{_i}'] = f'bid_vol_{_i}'

# ── OSS 大单分级阈值（按成交额，单位元）──
OSS_MEGA, OSS_LARGE, OSS_MID = 500000, 100000, 40000

# ── 有效交易时段（日内秒）：9:25 ~ 15:05 ──
SESSION_START = 9 * 3600 + 25 * 60      # 33900
SESSION_END = 15 * 3600 + 5 * 60        # 54300

# ── Task1：10 种交易模式 ──（名称, 解释, [(特征, 方向 +1/-1/0)]）
N_PATTERNS = 10
PATTERN_RULES = [
    ('大单吸筹', '资金大笔挂单买入，短时间内集中扫货',
     [('oss_mega_amount_pct', 1), ('ap_active_buy_pct', 1), ('pd_big_order_buy_ratio', 1),
      ('book_imbalance', 1), ('pi_time_concentration', 1), ('pd_reversal', -1)]),
    ('压单吸货', '卖盘挂大单压制股价，同时低位悄悄吸纳筹码',
     [('book_imbalance', -1), ('oss_large_amount_pct', 1), ('ap_active_buy_pct', 1),
      ('cb_sell_cancel_ratio', 1), ('pd_day_return', -1), ('oss_mega_amount_pct', 1)]),
    ('尾盘突袭', '尾盘集中放量拉升或砸盘，制造典型技术形态',
     [('pi_close10_amount_pct', 1), ('pd_close10_abs_return', 1), ('ap_unilateral_intensity', 1),
      ('pi_period_dispersion', -1), ('oss_large_amount_pct', 1), ('pd_big_order_direction', 1)]),
    ('集合竞价异动', '早盘阶段大幅拉高或压低，影响价格',
     [('pi_open30_amount_pct', 1), ('pd_open_change_pct', 1), ('ap_unilateral_intensity', 1),
      ('pd_reversal', 1), ('oss_mega_amount_pct', 1), ('pi_period_dispersion', -1)]),
    ('对倒拉升', '通过频繁大单成交制造成交量放大假象，配合拉升',
     [('ap_active_net_pct', 0), ('oss_large_amount_pct', 1), ('cb_cancel_ratio', 1),
      ('trd_total_count', 1), ('pd_day_return', 1), ('fd_herfindahl', 1)]),
    ('盘中诱多', '盘中快速拉升吸引跟风后回落',
     [('pd_reversal', 1), ('ap_active_buy_pct', 1), ('pd_day_return', -1),
      ('pi_time_concentration', -1), ('oss_mega_amount_pct', 1), ('pd_vwap_deviation', -1)]),
    ('分时脉冲', '短时间内快速拉升后迅速回落，试探上方抛压',
     [('rs_burst_ratio', 1), ('pd_high_low_pct', 1), ('ap_unilateral_intensity', 1),
      ('pi_period_dispersion', 1), ('rs_interval_cv', 1), ('pd_reversal', 1)]),
    ('涨停板打开', '封板后反复打开，制造换手假象',
     [('pd_day_return', 1), ('pd_high_low_pct', 1), ('trd_total_count', 1),
      ('pd_reversal', 1), ('ap_active_sell_pct', 1), ('fd_streak_max', -1)]),
    ('连续小单推升', '用大量小单持续买入缓慢推高股价，隐蔽建仓',
     [('oss_small_amount_pct', 1), ('ap_active_buy_pct', 1), ('rs_split_similarity', 1),
      ('pd_day_return', 1), ('oss_mega_amount_pct', -1), ('fd_streak_max', 1)]),
    ('日内套利', '资金在一定价格区间高频来回高抛低吸',
     [('oss_small_amount_pct', 1), ('rs_hft_ratio', 1), ('ap_active_net_pct', 0),
      ('cb_cancel_ratio', 1), ('rs_split_similarity', 1), ('fd_order_trade_ratio', 1)]),
]
PATTERN_NAMES = [p[0] for p in PATTERN_RULES]
PATTERN_DESC = {p[0]: p[1] for p in PATTERN_RULES}
PATTERN_CONDITIONS = {p[0]: p[2] for p in PATTERN_RULES}
PATTERN_RANGE = (0.06, 0.18)

# ── Task2：三分类 ──
CAPITAL_TYPES = ['游资', '量化', '散户']
CAPITAL_RANGE = (0.22, 0.45)
INTENTIONS = ['买入', '卖出', 'T0交易']

# Stage1 行为组聚类特征（量纲无关；trd_avg_trade_amount 在代码里 log1p）
BEHAVIOR_FEATURES = [
    'oss_small_amount_pct', 'pd_big_order_buy_ratio', 'rs_split_similarity', 'rs_hft_ratio',
    'cb_cancel_ratio', 'ap_unilateral_intensity', 'trd_avg_trade_amount', 'fd_herfindahl',
    'fd_order_trade_ratio', 'fd_buy_sell_asymmetry',
]

# 三因子加权特征（特征名, 权重）——值经 min-max→组内z-score 后加权
SCORE_WEIGHTS = {
    '游资': [('oss_mega_amount_pct', 0.12), ('pd_big_order_buy_ratio', 0.10), ('ap_unilateral_intensity', 0.10),
             ('fd_herfindahl', 0.09), ('fd_top5_amount_pct', 0.08), ('pi_time_concentration', 0.07),
             ('oss_large_amount_pct', 0.07), ('ap_active_net_pct', 0.06), ('book_imbalance', 0.06),
             ('pd_high_low_pct', 0.05), ('pd_vwap_deviation', 0.05), ('fd_streak_max', 0.05),
             ('pd_day_return', 0.05), ('rs_burst_ratio', 0.03)],
    '量化': [('rs_split_similarity', 0.12), ('cb_cancel_ratio', 0.11), ('rs_hft_ratio', 0.10),
             ('fd_order_trade_ratio', 0.09), ('oss_small_amount_pct', 0.08), ('cb_cancel_trade_ratio', 0.08),
             ('fd_cancel_burst_ratio', 0.07), ('oss_medium_amount_pct', 0.07), ('rs_burst_ratio', 0.06),
             ('trd_total_count', 0.06), ('fd_interval_autocorr', 0.05), ('fd_amount_entropy', 0.06)],
    '散户': [('oss_small_amount_pct', 0.15), ('rs_interval_cv', 0.13), ('spread', 0.11),
             ('obp_rel_spread_mean', 0.10), ('fd_buy_sell_asymmetry', 0.10), ('oss_small_count_pct', 0.10),
             ('rs_interval_kurt', 0.09), ('pi_period_dispersion', 0.08), ('pd_high_low_pct', 0.07),
             ('fd_amount_entropy', 0.07)],
}
