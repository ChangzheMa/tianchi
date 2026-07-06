"""通用多样性校准：把标签分布约束到 [tmin, tmax]（占比）。模式/资金复用。"""
import numpy as np
from collections import Counter


def calibrate_distribution(labels, score_matrix, class_names, tmin, tmax, rounds=8):
    """迭代再平衡。labels: (n,) 字符串数组；score_matrix: (n, len(class_names))。
    超比例类迁出"次优得分差距最小"的样本；欠比例类迁入"目标得分提升最大"的样本。"""
    labels = np.array(labels, dtype=object).copy()
    n = len(labels)
    idx_of = {c: i for i, c in enumerate(class_names)}
    lo, hi = int(np.floor(n * tmin)), int(np.ceil(n * tmax))

    for _ in range(rounds):
        cnt = Counter(labels)
        over = [c for c in class_names if cnt.get(c, 0) > hi]
        under = [c for c in class_names if cnt.get(c, 0) < lo]
        if not over and not under:
            break
        for c in over:
            ci = idx_of[c]
            excess = cnt[c] - hi
            members = np.where(labels == c)[0]
            alt = score_matrix[members].copy()
            alt[:, ci] = -np.inf
            best_alt = alt.max(axis=1)
            gap = score_matrix[members, ci] - best_alt
            for j in members[np.argsort(gap)[:excess]]:
                row = score_matrix[j].copy(); row[ci] = -np.inf
                labels[j] = class_names[int(np.argmax(row))]
            cnt = Counter(labels)
        for c in under:
            ci = idx_of[c]
            need = lo - cnt.get(c, 0)
            if need <= 0:
                continue
            cand = np.where(labels != c)[0]
            if len(cand) == 0:
                continue
            cur = score_matrix[cand, [idx_of[labels[k]] for k in cand]]
            gain = score_matrix[cand, ci] - cur
            for j in cand[np.argsort(-gain)[:need]]:
                labels[j] = c
            cnt = Counter(labels)
    return labels
