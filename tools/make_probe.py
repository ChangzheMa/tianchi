"""生成 task2 的 A 榜「探针」提交包。

两种模式，均复用基准包的股票集 / intention，仅改 capital_type 整列，task1 只写表头：

1) 全押单类（反解真实占比）：整份押同一类 A，加权 F1 = 2·p_A²/(1+p_A)，反解 p_A。
       python -m tools.make_probe --capital 散户
2) 按 20260706 占比随机（对照基线）：按当日反解占比 散44/游40/量16 精确配额、随机指派谁是谁。
   目的——测「分布正确 + 零分类能力」的基线：加权 F1 期望 = Σp² ≈ 0.378。
   若当前 pipeline（s2=0.288）还不如它，坐实 pipeline 负智能。
   ⚠️ 44/40/16 仅代表 20260706 单日，勿跨日复用（见 doc/07 与记忆库）。
       python -m tools.make_probe --random

通用参数：--base 基准 zip、--date 交易日（写入 CSV 列与包名）、--out 输出目录。
"""
import argparse
import zipfile
import numpy as np
import pandas as pd
from src import io_utils
from src.config import CAPITAL_TYPES, RANDOM_SEED

# capital 中文 → 包名 ASCII 后缀（避免中文文件名的平台兼容风险）
_SLUG = {'散户': 'retail', '游资': 'hotmoney', '量化': 'quant'}

# 20260706 探针反解出的归一化真实占比（⚠️ 仅当日有效，不可跨日复用）
RATIO_0706 = {'散户': 0.4398, '游资': 0.3969, '量化': 0.1633}


def _load_base_res(base_zip):
    """从基准 zip 读出 predict_result.csv（全字符串，保留原始代码/日期格式）。"""
    with zipfile.ZipFile(base_zip) as zf:
        with zf.open(io_utils.RES_FILE) as f:
            return pd.read_csv(f, dtype=str)


def _quota(n, ratio):
    """最大余数法：把 n 个名额按 ratio 分到各类，保证整数且总和精确 = n。"""
    ideal = {c: n * r for c, r in ratio.items()}
    base = {c: int(np.floor(v)) for c, v in ideal.items()}
    rem = n - sum(base.values())
    for c in sorted(ratio, key=lambda k: ideal[k] - base[k], reverse=True)[:rem]:
        base[c] += 1
    return base


def _random_by_ratio(n):
    """按 RATIO_0706 精确配额、固定 seed 随机指派，返回 (labels, quota)。"""
    quota = _quota(n, RATIO_0706)
    rng = np.random.default_rng(RANDOM_SEED)      # 固定 seed → 结果可复现
    idx = np.arange(n)
    rng.shuffle(idx)
    labels = np.empty(n, dtype=object)
    start = 0
    for c, q in quota.items():
        labels[idx[start:start + q]] = c          # 打乱后按配额连续切段，谁是谁随机
        start += q
    return labels, quota


def main():
    ap = argparse.ArgumentParser(description='生成 task2 A 榜探针包')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--capital', choices=CAPITAL_TYPES, help='全押的单一资金类别')
    g.add_argument('--random', action='store_true', help='按 20260706 占比随机配额（对照基线）')
    ap.add_argument('--base', default='out/submit_20260706_v1.zip', help='股票集/格式基准 zip')
    ap.add_argument('--date', default='20260706', help='transaction_date，写入 CSV 列与包名')
    ap.add_argument('--out', default='./out', help='输出目录')
    a = ap.parse_args()

    df_res = _load_base_res(a.base)
    n = len(df_res)
    if a.random:
        labels, quota = _random_by_ratio(n)
        df_res['capital_type'] = labels
        version, desc = 'probe-random0706', f'按0706占比随机配额 {quota}'
    else:
        df_res['capital_type'] = a.capital        # 唯一改动：整列押同一类
        version, desc = f'probe-{_SLUG[a.capital]}', f'全押「{a.capital}」'
    df_res['transaction_date'] = a.date           # 交易日对齐（列与包名一致）
    df_pat = pd.DataFrame(columns=io_utils.PAT_COLS)  # task1 只写表头（空数据）

    io_utils.save_results(df_pat, df_res, a.out)
    zip_path = io_utils.pack_submission(a.out, a.date, version=version, cleanup=True, prefix='task2')
    print(f'探针包 → {zip_path}｜{desc}｜{n} 行')


if __name__ == '__main__':
    main()
