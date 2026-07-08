"""生成「全押单一 capital_type」探针提交包，用于反解 A 榜真实类分布。

原理：若整份 predict_result.csv 全押同一类 A，则 sklearn 加权 F1 有闭式解
      weightedF1 = 2·p_A² / (1 + p_A)（p_A = 该类真实占比，单调可反解）。
      分别全押三类各提一次，即可解出官方标签的真实三类分布。

控制变量：复用基准包的股票集 / intention，仅改 capital_type 整列 + transaction_date，
          task1（pattern_reco.csv）只写表头。这样与基准 s2 对照干净。

用法：
    python -m tools.make_probe --capital 散户                 # 默认基于 out/submit_20260706_v1.zip
    python -m tools.make_probe --capital 游资 --date 20260707 # 改交易日（同时写入 CSV 列与包名）
"""
import argparse
import zipfile
import pandas as pd
from src import io_utils
from src.config import CAPITAL_TYPES

# capital 中文 → 包名用的 ASCII 后缀（避免中文文件名在平台上的兼容风险）
_SLUG = {'散户': 'retail', '游资': 'hotmoney', '量化': 'quant'}


def _load_base_res(base_zip):
    """从基准 zip 读出 predict_result.csv（全部当字符串，保留原始代码/日期格式）。"""
    with zipfile.ZipFile(base_zip) as zf:
        with zf.open(io_utils.RES_FILE) as f:
            return pd.read_csv(f, dtype=str)


def main():
    ap = argparse.ArgumentParser(description='生成全押单类的 A 榜分布探针包')
    ap.add_argument('--capital', required=True, choices=CAPITAL_TYPES, help='全押的资金类别')
    ap.add_argument('--base', default='out/submit_20260706_v1.zip', help='作为股票集/格式基准的既有 zip')
    ap.add_argument('--date', default='20260706', help='transaction_date，同时写入 CSV 列与包名')
    ap.add_argument('--out', default='./out', help='输出目录')
    a = ap.parse_args()

    df_res = _load_base_res(a.base)
    df_res['capital_type'] = a.capital          # 唯一实质改动：整列押同一类
    df_res['transaction_date'] = a.date         # 交易日对齐（列与包名一致）
    df_pat = pd.DataFrame(columns=io_utils.PAT_COLS)  # task1 只写表头（空数据）

    io_utils.save_results(df_pat, df_res, a.out)
    zip_path = io_utils.pack_submission(
        a.out, a.date, version=f'probe-{_SLUG[a.capital]}', cleanup=True, prefix='task2')
    print(f'探针包 → {zip_path}｜全押「{a.capital}」｜{len(df_res)} 行')


if __name__ == '__main__':
    main()
