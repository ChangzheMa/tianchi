"""Task2：行为组聚类 → 组内 Z-score → 游资/量化/散户三因子打分 + 简版意图。"""
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from src.config import (RANDOM_SEED, CAPITAL_TYPES, SCORE_WEIGHTS, BEHAVIOR_FEATURES, INTENTIONS)

EPS = 1e-8


def behavior_groups(df_feat):
    """在量纲无关行为特征上 KMeans 聚 3~6 组。"""
    feats = [c for c in BEHAVIOR_FEATURES if c in df_feat.columns]
    mat = df_feat[feats].values.astype(float).copy()
    if 'trd_avg_trade_amount' in feats:
        j = feats.index('trd_avg_trade_amount')
        mat[:, j] = np.log1p(np.maximum(mat[:, j], 0))
    mat = np.nan_to_num(mat)
    n = len(df_feat)
    if n < 3:
        return np.zeros(n, dtype=int)
    mat = RobustScaler().fit_transform(mat)
    k = min(6, max(3, n // 100))
    k = min(k, n)
    return KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_SEED).fit_predict(mat)


def _normalize_score_feats(df_feat, groups):
    """全局 min-max → 组内 z-score。返回 {feat: ndarray}。"""
    feats = sorted({c for ws in SCORE_WEIGHTS.values() for c, _ in ws})
    norm = {}
    for c in feats:
        v = np.nan_to_num(df_feat[c].values.astype(float)) if c in df_feat.columns else np.zeros(len(df_feat))
        mn, mx = v.min(), v.max()
        norm[c] = (v - mn) / (mx - mn + EPS) if mx > mn else np.full(len(v), 0.5)
    for g in np.unique(groups):
        gm = groups == g
        if gm.sum() < 3:
            continue
        for c in feats:
            gv = norm[c][gm]
            sd = gv.std()
            norm[c][gm] = (gv - gv.mean()) / (sd + EPS) if sd > EPS else 0.0
    return norm


def compute_capital_scores(df_feat, groups):
    """三因子加权得分，列序 = CAPITAL_TYPES。"""
    norm = _normalize_score_feats(df_feat, groups)
    n = len(df_feat)
    scores = np.zeros((n, len(CAPITAL_TYPES)))
    for ci, ctype in enumerate(CAPITAL_TYPES):
        s = np.zeros(n)
        wsum = sum(w for _, w in SCORE_WEIGHTS[ctype]) + EPS
        for feat, w in SCORE_WEIGHTS[ctype]:
            s += norm.get(feat, np.zeros(n)) * w
        scores[:, ci] = s / wsum
    return scores


def judge_intention(df_feat):
    """简版意图：净买卖比 + 盘口失衡 + VWAP偏离 + 日内收益 → 买入/卖出/T0交易。"""
    def col(c):
        return df_feat[c].values.astype(float) if c in df_feat.columns else np.zeros(len(df_feat))
    net = col('ap_active_net_pct')
    imb = col('obp_imbalance_mean')
    vdev = col('pd_vwap_deviation')
    ret = col('pd_day_return')
    buy = (net > 0.01).astype(int) + (imb > 0.02).astype(int) + (vdev > 0.002).astype(int) + (ret > 0.005).astype(int)
    sell = (net < -0.01).astype(int) + (imb < -0.02).astype(int) + (vdev < -0.002).astype(int) + (ret < -0.005).astype(int)
    out = np.full(len(df_feat), 'T0交易', dtype=object)
    out[(buy >= 2) & (buy > sell)] = '买入'
    out[(sell >= 2) & (sell > buy)] = '卖出'
    return out
