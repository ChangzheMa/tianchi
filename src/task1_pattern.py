"""Task1：RobustScaler+Rank 混合归一化 → KMeans 聚类 → PATTERN_RULES 贪心映射。"""
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from src.config import (RANDOM_SEED, N_PATTERNS, PATTERN_NAMES, PATTERN_CONDITIONS)

EPS = 1e-8
_META = ('stock_code', 'transaction_date', 'pattern_type', 'cluster_id')


def _feat_cols(df):
    return [c for c in df.columns if c not in _META and
            df[c].dtype.kind in 'fi']


def normalize_robust(df_feat):
    """50% RobustScaler + 50% Rank 百分位混合。返回 (矩阵, 特征列名)。"""
    cols = _feat_cols(df_feat)
    mat = np.nan_to_num(df_feat[cols].values.astype(float), nan=0, posinf=0, neginf=0)
    scaled = RobustScaler().fit_transform(mat)
    n, d = scaled.shape
    rank = np.zeros((n, d))
    for j in range(d):
        rank[:, j] = np.argsort(np.argsort(scaled[:, j])) / (n - 1) if n > 1 else 0.5
    hybrid = 0.5 * scaled / (np.std(scaled, axis=0) + EPS) + 0.5 * rank
    return np.nan_to_num(hybrid), cols


def run_clustering(feat_hybrid, k):
    """KMeans++；样本数 < k 时自动降 k。"""
    k = max(1, min(k, feat_hybrid.shape[0]))
    km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                random_state=RANDOM_SEED, max_iter=300)
    return km.fit_predict(feat_hybrid)


def compute_pattern_scores(df_feat):
    """每样本对每种模式按跨样本百分位方向打分。返回 (n, N_PATTERNS)。"""
    n = len(df_feat)
    scores = np.zeros((n, N_PATTERNS))
    for pi, pname in enumerate(PATTERN_NAMES):
        conds = PATTERN_CONDITIONS[pname]
        s = np.zeros(n)
        for col, direction in conds:
            vals = df_feat[col].values.astype(float) if col in df_feat.columns else np.zeros(n)
            order = np.argsort(np.argsort(vals))
            pct = order / (n - 1) if n > 1 else np.full(n, 0.5)
            if direction == 1:
                s += pct
            elif direction == -1:
                s += (1 - pct)
            else:
                s += 1 - np.abs(pct - 0.5) * 2
        scores[:, pi] = s / len(conds)
    return scores


def map_clusters_to_patterns(labels, pattern_scores):
    """按簇大小排序，每簇贪心分配得分最高且未占用的模式；返回每样本模式名数组。"""
    result = np.empty(len(labels), dtype=object)
    uniq = sorted(set(labels), key=lambda l: -(labels == l).sum())
    assigned = set()
    for lab in uniq:
        mask = labels == lab
        mean_scores = pattern_scores[mask].mean(axis=0)
        for pi in np.argsort(-mean_scores):
            if pi not in assigned:
                assigned.add(pi)
                result[mask] = PATTERN_NAMES[pi]
                break
        else:  # 模式已用尽（簇数 > 10 时）
            result[mask] = PATTERN_NAMES[int(np.argmax(mean_scores))]
    return result
