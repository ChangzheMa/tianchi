"""股票样本读取、结果格式校验与合法输出（UTF-8 无 BOM + LF，去代码后缀）。"""
import os
import re
import zipfile
import pandas as pd
from src.config import CAPITAL_TYPES, INTENTIONS

PAT_COLS = ['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']
RES_COLS = ['stock_code', 'transaction_date', 'capital_type', 'capital_intention']
PAT_FILE = 'pattern_reco.csv'
RES_FILE = 'predict_result.csv'


def _strip_suffix(code):
    return re.sub(r'\.(SH|SZ)$', '', str(code), flags=re.IGNORECASE)


def load_stock_sample(path):
    """读 stock_sample.csv，返回去后缀的代码集合。"""
    df = pd.read_csv(path, dtype=str)
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
    df_pat.to_csv(os.path.join(out_dir, PAT_FILE),
                  index=False, encoding='utf-8', lineterminator='\n')
    df_res.to_csv(os.path.join(out_dir, RES_FILE),
                  index=False, encoding='utf-8', lineterminator='\n')
    print(f'已保存 {PAT_FILE} / {RES_FILE} 到 {out_dir}')


def pack_submission(out_dir, date_tag, version=None, cleanup=False, prefix='submit'):
    """把两个结果 CSV 打包为 <prefix>_<date_tag>[_<version>].zip（zip 根目录，无路径层级）。
    prefix 缺省 submit；单任务模式传 task1/task2 → taskN_<日期>_<版本>.zip。
    cleanup=True 时打包完删除源 CSV，只保留 zip。返回 zip 路径。"""
    suffix = f'_{version}' if version else ''
    zip_path = os.path.join(out_dir, f'{prefix}_{date_tag}{suffix}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in (PAT_FILE, RES_FILE):
            fp = os.path.join(out_dir, fn)
            if not os.path.exists(fp):
                raise FileNotFoundError(fp)
            zf.write(fp, arcname=fn)          # arcname 仅文件名 → zip 内无子目录
    if cleanup:
        for fn in (PAT_FILE, RES_FILE):
            fp = os.path.join(out_dir, fn)
            if os.path.exists(fp):
                os.remove(fp)
        print(f'已打包 {zip_path}（已清理中间 CSV）')
    else:
        print(f'已打包 {zip_path}')
    return zip_path
