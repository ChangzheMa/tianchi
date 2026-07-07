"""10 大类约 60+ 维特征工程。每只股票单日聚合为一行 dict。"""
import numpy as np
import pandas as pd
from src import data_io
from src.config import (OSS_MEGA, OSS_LARGE, OSS_MID, SESSION_START, SESSION_END,
                        PATTERN_CONDITIONS, SCORE_WEIGHTS, BEHAVIOR_FEATURES,
                        ORDER_CANCEL)

EPS = 1e-8


def _safe(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def _all_referenced_features():
    ref = {c for conds in PATTERN_CONDITIONS.values() for c, _ in conds}
    ref |= {c for ws in SCORE_WEIGHTS.values() for c, _ in ws}
    ref |= set(BEHAVIOR_FEATURES)
    return ref


def extract_features(stock_code, date_str, base_dir='.'):
    """返回单只股票单日特征字典。空/退化数据填 0。"""
    f = {'stock_code': str(stock_code), 'transaction_date': str(date_str)}
    trd = data_io.load_trades(stock_code, date_str, base_dir)
    ordf = data_io.load_orders(stock_code, date_str, base_dir)
    hq = data_io.load_snapshot(stock_code, date_str, base_dir)

    trd = trd[(trd['seconds'] >= SESSION_START) & (trd['seconds'] <= SESSION_END)]
    hq = hq[(hq['seconds'] >= SESSION_START) & (hq['seconds'] <= SESSION_END)]
    n = len(trd)
    ta = trd['amount'].sum() + EPS

    # ── OSS 大单分级（按成交额）──
    amt = trd['amount'] if n else pd.Series([], dtype=float)
    masks = {
        'oss_mega': amt >= OSS_MEGA,
        'oss_large': (amt >= OSS_LARGE) & (amt < OSS_MEGA),
        'oss_medium': (amt >= OSS_MID) & (amt < OSS_LARGE),
        'oss_small': amt < OSS_MID,
    }
    for key, mask in masks.items():
        f[f'{key}_amount_pct'] = amt[mask].sum() / ta if n else 0.0
        f[f'{key}_count_pct'] = mask.sum() / n if n else 0.0

    # ── TRD 交易结构 ──
    f['trd_avg_trade_amount'] = ta / n if n else 0.0
    f['trd_avg_trade_volume'] = (trd['volume'].sum() / n) if n else 0.0
    f['trd_trade_amount_std'] = _safe(amt.std()) if n else 0.0
    f['trd_total_count'] = float(n)

    # ── RS 订单时序 ──
    iv = trd['seconds'].diff().dropna()
    if len(iv) > 1:
        cv = _safe(iv.std() / (iv.mean() + EPS))
        f['rs_interval_cv'] = cv
        f['rs_split_similarity'] = max(0.0, 1 - cv)
        f['rs_burst_ratio'] = (iv < 1).sum() / len(iv)
        f['rs_hft_ratio'] = (iv < 0.5).sum() / len(iv)
        z = (iv - iv.mean()) / (iv.std() + EPS)
        f['rs_interval_skew'] = _safe((z ** 3).mean())
        f['rs_interval_kurt'] = _safe((z ** 4).mean() - 3)
    else:
        for k in ['rs_interval_cv', 'rs_split_similarity', 'rs_burst_ratio', 'rs_hft_ratio',
                  'rs_interval_skew', 'rs_interval_kurt']:
            f[k] = 0.0

    # ── CB 撤单行为 ──
    no = len(ordf)
    if no:
        cancel = ordf['order_type'] == ORDER_CANCEL
        n_cancel = int(cancel.sum())
        f['cb_cancel_ratio'] = n_cancel / no
        f['cb_trade_ord_ratio'] = (ordf['order_type'] == '1').sum() / no
        f['cb_cancel_amount_ratio'] = ordf.loc[cancel, 'volume'].sum() / (ordf['volume'].sum() + EPS)
        f['cb_cancel_trade_ratio'] = n_cancel / (n + 1)
        cb = ordf[cancel]
        f['cb_buy_cancel_ratio'] = (cb['side'] == 'B').sum() / (n_cancel + EPS)
        f['cb_sell_cancel_ratio'] = (cb['side'] == 'S').sum() / (n_cancel + EPS)
    else:
        for k in ['cb_cancel_ratio', 'cb_trade_ord_ratio', 'cb_cancel_amount_ratio',
                  'cb_cancel_trade_ratio', 'cb_buy_cancel_ratio', 'cb_sell_cancel_ratio']:
            f[k] = 0.0

    # ── AP 主动成交（BS 标志）──
    if n:
        bm, sm = trd['side'] == 'B', trd['side'] == 'S'
        ba, sa = trd.loc[bm, 'amount'].sum(), trd.loc[sm, 'amount'].sum()
        at = ba + sa + EPS
        f['ap_active_buy_pct'] = ba / at
        f['ap_active_sell_pct'] = sa / at
        f['ap_active_net_pct'] = (ba - sa) / ta
        f['ap_unilateral_intensity'] = abs(f['ap_active_net_pct'])
        import itertools
        runs_b = [len(list(g)) for k_, g in itertools.groupby(trd['side'] == 'B') if k_]
        runs_s = [len(list(g)) for k_, g in itertools.groupby(trd['side'] == 'S') if k_]
        f['ap_active_buy_run_max'] = float(max(runs_b) if runs_b else 0)
        f['ap_active_sell_run_max'] = float(max(runs_s) if runs_s else 0)
    else:
        for k in ['ap_active_buy_pct', 'ap_active_sell_pct', 'ap_active_net_pct',
                  'ap_unilateral_intensity', 'ap_active_buy_run_max', 'ap_active_sell_run_max']:
            f[k] = 0.0

    # ── PI 日内时段 ──
    def _pct(lo, hi):
        m = (trd['seconds'] >= lo) & (trd['seconds'] < hi)
        return trd.loc[m, 'amount'].sum() / ta if n else 0.0
    f['pi_open30_amount_pct'] = _pct(9 * 3600 + 30 * 60, 10 * 3600)
    f['pi_close10_amount_pct'] = _pct(14 * 3600 + 50 * 60, 15 * 3600)
    f['pi_time_concentration'] = f['pi_open30_amount_pct'] + f['pi_close10_amount_pct']
    bins = [_pct(9 * 3600 + 30 * 60 + i * 1800, 9 * 3600 + 30 * 60 + (i + 1) * 1800) for i in range(8)]
    f['pi_period_dispersion'] = _safe(np.std(bins))

    # ── PD 价格发现（行情快照）──
    if len(hq):
        op = _safe(hq['open'].iloc[0]); pc = _safe(hq['prev_close'].iloc[0])
        cl = _safe(hq['price'].iloc[-1]); hi = _safe(hq['high'].max()); lo = _safe(hq['low'].min())
        vwap = _safe((hq['price'] * hq['volume']).sum() / (hq['volume'].sum() + EPS))
        f['pd_open_change_pct'] = (op - pc) / (pc + EPS)
        f['pd_day_return'] = (cl - op) / (op + EPS)
        f['pd_high_low_pct'] = (hi - lo) / (op + EPS)
        f['pd_vwap_deviation'] = (cl - vwap) / (vwap + EPS)
        half = len(hq) // 2
        m_dir = np.sign(hq['price'].iloc[half] - hq['price'].iloc[0]) if half else 0
        a_dir = np.sign(hq['price'].iloc[-1] - hq['price'].iloc[half]) if half else 0
        f['pd_reversal'] = 1.0 if m_dir != a_dir else 0.0
        close10 = hq[hq['seconds'] >= 14 * 3600 + 50 * 60]
        f['pd_close10_abs_return'] = abs(_safe((close10['price'].iloc[-1] - close10['price'].iloc[0])
                                               / (close10['price'].iloc[0] + EPS))) if len(close10) > 1 else 0.0
    else:
        for k in ['pd_open_change_pct', 'pd_day_return', 'pd_high_low_pct', 'pd_vwap_deviation',
                  'pd_reversal', 'pd_close10_abs_return']:
            f[k] = 0.0
    big = trd[trd['amount'] >= OSS_LARGE] if n else trd
    if len(big):
        bb = (big['side'] == 'B').sum() / len(big)
        f['pd_big_order_buy_ratio'] = bb
        f['pd_big_order_amount_pct'] = big['amount'].sum() / ta
        f['pd_big_order_direction'] = abs(bb - 0.5) * 2
    else:
        f['pd_big_order_buy_ratio'] = 0.0
        f['pd_big_order_amount_pct'] = 0.0
        f['pd_big_order_direction'] = 0.0

    # ── OBP 盘口衍生 + 盘口动态 ──
    if len(hq):
        ap1 = hq['ask_price_1'].values; bp1 = hq['bid_price_1'].values
        av1 = hq['ask_vol_1'].values; bv1 = hq['bid_vol_1'].values
        spreads = ap1 - bp1
        f['spread'] = _safe(np.nanmean(spreads))
        f['obp_spread_std'] = _safe(np.nanstd(spreads))
        f['obp_rel_spread_mean'] = _safe(np.nanmean(spreads / (hq['price'].values + EPS)))
        f['book_imbalance'] = _safe(np.nanmean((bv1 - av1) / (bv1 + av1 + EPS)))
        tb, tav = hq['total_bid_vol'].values, hq['total_ask_vol'].values
        imb = (tb - tav) / (tb + tav + EPS)
        f['obp_imbalance_mean'] = _safe(np.nanmean(imb))
        f['obp_imbalance_std'] = _safe(np.nanstd(imb))
        f['obp_weighted_spread_mean'] = _safe(np.nanmean(hq['weighted_ask'].values - hq['weighted_bid'].values))
        f['obp_bid_ask_ratio'] = _safe(np.nanmean(tb) / (np.nanmean(tav) + EPS))
        f['obp_depth_change'] = _safe((bv1[-1] - bv1[0]) / (bv1[0] + EPS))
        f['obp_net_depth_change'] = _safe(((bv1[-1] - bv1[0]) - (av1[-1] - av1[0])) / (bv1[0] + av1[0] + EPS))
        big_bid = sum(hq[f'bid_vol_{i}'].iloc[0] for i in range(1, 4))
        f['big_bid_ratio'] = _safe(big_bid / (tb[0] + EPS))
    else:
        for k in ['spread', 'obp_spread_std', 'obp_rel_spread_mean', 'book_imbalance',
                  'obp_imbalance_mean', 'obp_imbalance_std', 'obp_weighted_spread_mean',
                  'obp_bid_ask_ratio', 'obp_depth_change', 'obp_net_depth_change', 'big_bid_ratio']:
            f[k] = 0.0

    # ── 资金集中度 ──
    if n:
        amt_pct = (trd['amount'] / ta).values
        f['fd_herfindahl'] = _safe(np.sum(amt_pct ** 2))
        top5 = np.sort(trd['amount'].values)[-max(1, n // 20):]
        f['fd_top5_amount_pct'] = _safe(top5.sum() / ta)
        f['fd_order_trade_ratio'] = no / (n + EPS)
        f['fd_buy_sell_asymmetry'] = abs((trd['side'] == 'B').sum() - (trd['side'] == 'S').sum()) / n
        p = amt_pct[amt_pct > 0]
        f['fd_amount_entropy'] = _safe(-np.sum(p * np.log(p + EPS)))
        if len(iv) > 2:
            f['fd_interval_autocorr'] = _safe(np.corrcoef(iv.values[:-1], iv.values[1:])[0, 1])
        else:
            f['fd_interval_autocorr'] = 0.0
        import itertools
        streaks = [len(list(g)) for _, g in itertools.groupby(trd['side'])]
        f['fd_streak_max'] = float(max(streaks) if streaks else 0)
    else:
        for k in ['fd_herfindahl', 'fd_top5_amount_pct', 'fd_order_trade_ratio',
                  'fd_buy_sell_asymmetry', 'fd_amount_entropy', 'fd_interval_autocorr', 'fd_streak_max']:
            f[k] = 0.0
    if no:
        civ = ordf.loc[ordf['order_type'] == ORDER_CANCEL, 'seconds'].diff().dropna()
        f['fd_cancel_burst_ratio'] = (civ < 1).sum() / len(civ) if len(civ) else 0.0
    else:
        f['fd_cancel_burst_ratio'] = 0.0

    # 兜底：config 引用但上面漏产出的特征补 0（防御性）
    for k in _all_referenced_features():
        f.setdefault(k, 0.0)
    return f


def build_feature_matrix(pairs, base_dir='.'):
    """对 (股票,日期) 列表逐个提特征，缺文件跳过并告警。"""
    rows = []
    for sc, td in pairs:
        try:
            rows.append(extract_features(sc, td, base_dir))
        except FileNotFoundError as e:
            print(f'[warn] 跳过 {sc}/{td}：缺文件 {e}')
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ('stock_code', 'transaction_date')]
    df[num] = df[num].replace([np.inf, -np.inf], 0.0)
    for c in num:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    return df.fillna(0.0)
