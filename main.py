"""AFAC2026 赛题一 Baseline（精简忠实版）入口。

流程：三源CSV → 特征矩阵 → Task1聚类+模式映射 → Task2三因子资金识别
      → 分布校准 → 打包 submit_<日期>_<版本>.zip（中间 CSV 打包后删除，仅留 zip）
运行：python main.py                         # 默认：全部日期 × 过滤到 stock_sample.csv 的100只
      python main.py -d 20260703 -o ./out     # 指定日期/输出
      python main.py --sample all             # 不过滤，分析该日期全部股票
      python main.py -n 20                     # 每日限跑前N只（调试）
"""
import os
import argparse
import numpy as np
import pandas as pd
from src import features, task1_pattern, task2_capital, calibrate, io_utils
from src.config import (N_PATTERNS, PATTERN_NAMES, PATTERN_DESC, PATTERN_RANGE,
                        CAPITAL_TYPES, CAPITAL_RANGE, VERSION)


def _discover_pairs(data_dir, dates, limit):
    """枚举 data_dir/<date>/<stock> → [(stock, date)]。"""
    pairs = []
    all_dates = dates or sorted(d for d in os.listdir(data_dir)
                                if os.path.isdir(os.path.join(data_dir, d)))
    for date in all_dates:
        ddir = os.path.join(data_dir, date)
        if not os.path.isdir(ddir):
            continue
        stocks = sorted(s for s in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, s)))
        if limit:
            stocks = stocks[:limit]
        pairs += [(s, date) for s in stocks]
    return pairs


def run(data_dir='./data', dates=None, out_dir='./out', sample=None, limit=None):
    """执行完整流水线。"""
    pairs = _discover_pairs(data_dir, dates, limit)
    if not pairs:
        raise SystemExit(f'未在 {data_dir} 下发现数据（<日期>/<股票>/三源CSV）')
    print(f'[1/4] 提取特征：{len(pairs)} 个(股票,日期)')
    m = features.build_feature_matrix(pairs, data_dir)

    print('[2/4] Task1 聚类与模式映射')
    arr, _ = task1_pattern.normalize_robust(m)
    labels = task1_pattern.run_clustering(arr, k=min(N_PATTERNS, len(m)))
    pat_scores = task1_pattern.compute_pattern_scores(m)
    pat = task1_pattern.map_clusters_to_patterns(labels, pat_scores)
    pat = calibrate.calibrate_distribution(pat, pat_scores, PATTERN_NAMES, *PATTERN_RANGE)

    print('[3/4] Task2 资金识别与意图')
    groups = task2_capital.behavior_groups(m)
    cap_scores = task2_capital.compute_capital_scores(m, groups)
    cap = calibrate.calibrate_distribution(
        np.array([CAPITAL_TYPES[i] for i in cap_scores.argmax(axis=1)], dtype=object),
        cap_scores, CAPITAL_TYPES, *CAPITAL_RANGE)
    intention = task2_capital.judge_intention(m)

    df_pat = m[['stock_code', 'transaction_date']].copy()
    df_pat['pattern_type'] = pat
    df_pat['pattern_explanation'] = [PATTERN_DESC[p] for p in pat]
    df_res = m[['stock_code', 'transaction_date']].copy()
    df_res['capital_type'] = cap
    df_res['capital_intention'] = intention

    if sample and sample != 'all':                         # sample='all' → 不过滤，分析全部
        if not os.path.exists(sample):
            raise SystemExit(f'样本文件 {sample} 不存在；用 --sample all 分析全部股票，或指定正确路径')
        keep = io_utils.load_stock_sample(sample)          # 去后缀的代码集合
        _sc = df_pat['stock_code'].map(io_utils._strip_suffix)  # 目录名带 .SH，比对前先剥离
        df_pat = df_pat[_sc.isin(keep)]
        df_res = df_res[df_res['stock_code'].map(io_utils._strip_suffix).isin(keep)]

    print('[4/4] 校验与保存')
    io_utils.save_results(df_pat, df_res, out_dir)
    dates = sorted(df_res['transaction_date'].astype(str).unique())
    tag = dates[0] if len(dates) == 1 else f'{dates[0]}-{dates[-1]}'
    if len(dates) > 1:
        print(f'[warn] 结果含多个交易日 {dates}，submit 文件名用日期范围 {tag}')
    io_utils.pack_submission(out_dir, tag, version=VERSION, cleanup=True)
    print(f'模式分布:\n{pd.Series(pat).value_counts().to_string()}')
    print(f'资金分布:\n{pd.Series(cap).value_counts().to_string()}')
    return df_pat, df_res


def main():
    ap = argparse.ArgumentParser(description='AFAC2026 赛题一 Baseline（精简忠实版）')
    ap.add_argument('--data', default='./data', help='数据根目录')
    ap.add_argument('--date', '-d', nargs='*', default=None, help='指定日期(可多值)，缺省跑全部')
    ap.add_argument('--out', '-o', default='./out', help='输出目录')
    ap.add_argument('--sample', default='stock_sample.csv',
                    help="样本CSV路径，过滤到目标股票；传 all 分析该日期全部股票（默认 stock_sample.csv）")
    ap.add_argument('-n', type=int, default=None, help='每日限跑前 N 只（调试）')
    a = ap.parse_args()
    run(data_dir=a.data, dates=a.date, out_dir=a.out, sample=a.sample, limit=a.n)


if __name__ == '__main__':
    main()
