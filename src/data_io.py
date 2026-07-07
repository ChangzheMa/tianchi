"""三源 Level-2 CSV 读取（GBK 中文列名 → 英文，时间标准化）。"""
import os
import numpy as np
import pandas as pd
from src.config import HQ_COL_MAP, HQ_PRICE_COLS, PRICE_SCALE


def parse_time_to_seconds(time_val):
    """HHMMSSmmm → 日内秒数。支持标量或数组。"""
    t = np.asarray(time_val, dtype=np.int64)
    return (t // 10000000) * 3600 + ((t // 100000) % 100) * 60 + ((t // 1000) % 100)


def _path(stock_code, date_str, base_dir, fname):
    p = os.path.join(base_dir, date_str, str(stock_code), fname)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def load_snapshot(stock_code, date_str, base_dir='.'):
    """加载 3 秒行情快照，中文列名映射为英文，按 seconds 排序。"""
    df = pd.read_csv(_path(stock_code, date_str, base_dir, '行情.csv'),
                     encoding='gbk', low_memory=False)
    df = df.rename(columns=HQ_COL_MAP)
    for c in HQ_PRICE_COLS:                          # 价格列还原为元（÷ PRICE_SCALE）
        if c in df.columns:
            df[c] = df[c] / PRICE_SCALE
    df['seconds'] = parse_time_to_seconds(df['time'].values)
    df['stock_code'] = str(stock_code)
    df['transaction_date'] = str(date_str)
    return df.sort_values('seconds').reset_index(drop=True)


def load_trades(stock_code, date_str, base_dir='.'):
    """加载逐笔成交，amount = 成交价格 × 成交数量。"""
    df = pd.read_csv(_path(stock_code, date_str, base_dir, '逐笔成交.csv'),
                     encoding='gbk', low_memory=False)
    df = df.rename(columns={'时间': 'time', 'BS标志': 'side',
                            '成交价格': 'price', '成交数量': 'volume'})
    df['price'] = df['price'] / PRICE_SCALE         # 还原为元
    df['seconds'] = parse_time_to_seconds(df['time'].values)
    df['amount'] = df['price'] * df['volume']       # 成交额（元）
    return df.sort_values('seconds').reset_index(drop=True)


def load_orders(stock_code, date_str, base_dir='.'):
    """加载逐笔委托（SSE：委托类型 A新增/D撤单；委托代码 B买/S卖）。"""
    df = pd.read_csv(_path(stock_code, date_str, base_dir, '逐笔委托.csv'),
                     encoding='gbk', low_memory=False)
    df = df.rename(columns={'时间': 'time', '委托类型': 'order_type', '委托代码': 'side',
                            '委托价格': 'price', '委托数量': 'volume'})
    df['order_type'] = df['order_type'].astype(str)
    df['price'] = df['price'] / PRICE_SCALE         # 还原为元
    df['seconds'] = parse_time_to_seconds(df['time'].values)
    return df.sort_values('seconds').reset_index(drop=True)
