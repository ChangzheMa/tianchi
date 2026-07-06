"""股票样本读取、结果格式校验与合法输出（UTF-8 无 BOM + LF，去代码后缀）。"""
import os
import re
import pandas as pd
from src.config import CAPITAL_TYPES, INTENTIONS

PAT_COLS = ['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']
RES_COLS = ['stock_code', 'transaction_date', 'capital_type', 'capital_intention']


def _strip_suffix(code):
    return re.sub(r'\.(SH|SZ)$', '', str(code), flags=re.IGNORECASE)


def load_stock_sample(path):
    """读 股票样本.xlsx，返回去后缀的代码集合。"""
    df = pd.read_excel(path, engine='openpyxl', dtype=str)
    col = next((c for c in df.columns if df[c].astype(str).str.contains(r'\d{6}').any()), df.columns[0])
    return {_strip_suffix(x) for x in df[col].dropna()}


def save_results(df_pat, df_res, out_dir):
    """校验字段/合法值 → 去后缀 → UTF-8(无BOM)+LF 写盘。"""
    df_pat = df_pat[PAT_COLS].copy()
    df_res = df_res[RES_COLS].copy()
    assert list(df_pat.columns) == PAT_COLS
    assert list(df_res.columns) == RES_COLS
    assert df_res['capital_type'].isin(CAPITAL_TYPES).all(), \
        f'非法 capital_type: {set(df_res["capital_type"]) - set(CAPITAL_TYPES)}'
    assert df_res['capital_intention'].isin(INTENTIONS).all(), \
        f'非法 capital_intention: {set(df_res["capital_intention"]) - set(INTENTIONS)}'
    df_pat['stock_code'] = df_pat['stock_code'].map(_strip_suffix)
    df_res['stock_code'] = df_res['stock_code'].map(_strip_suffix)
    os.makedirs(out_dir, exist_ok=True)
    df_pat.to_csv(os.path.join(out_dir, 'pattern_reco.csv'),
                  index=False, encoding='utf-8', lineterminator='\n')
    df_res.to_csv(os.path.join(out_dir, 'predict_result.csv'),
                  index=False, encoding='utf-8', lineterminator='\n')
    print(f'已保存 pattern_reco.csv / predict_result.csv 到 {out_dir}')
