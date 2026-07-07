"""合成三源 Level-2 CSV 测试数据（GBK / 中文列名 / HHMMSSmmm）。

三种行为原型：游资（大单集中单边）/ 量化（高频小单多撤单）/ 散户（零散不规则）。
用作 TDD 与端到端冒烟；真实数据到位后直接替换 data/ 下内容即可。
"""
import os
import numpy as np
import pandas as pd
from src.config import PRICE_SCALE, ORDER_ADD, ORDER_CANCEL

RANDOM_SEED = 42
ARCHETYPES = ['游资', '量化', '散户']

# 需 × PRICE_SCALE 还原为交易所原始量纲的价格列（模拟真实数据；金额/量列保持不动）
_PRICE_COLS_CN = (['成交价', '开盘价', '前收盘', '最高价', '最低价',
                   '加权平均叫卖价', '加权平均叫买价']
                  + [f'申卖价{i}' for i in range(1, 11)]
                  + [f'申买价{i}' for i in range(1, 11)])


def _scale_price(df, cols):
    """把元价格列 × PRICE_SCALE 转整数，模拟真实交易所原始量纲。"""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (df[c] * PRICE_SCALE).round().astype('int64')
    return df


def _sec_to_hhmmssmmm(sec):
    """日内秒 → HHMMSSmmm 整数。"""
    sec = np.asarray(sec, dtype=np.int64)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return (h * 10000000 + m * 100000 + s * 1000).astype(np.int64)


def _gen_trades(rng, archetype, base_price):
    """按原型生成逐笔成交 (时间秒, side, 价, 量)。"""
    if archetype == '游资':
        n = rng.integers(400, 600)
        vol = rng.integers(2000, 20000, n)
        side = rng.choice(['B', 'S'], n, p=[0.72, 0.28])
    elif archetype == '量化':
        n = rng.integers(2500, 4000)
        vol = rng.integers(100, 800, n)
        side = rng.choice(['B', 'S'], n, p=[0.5, 0.5])
    else:
        n = rng.integers(600, 1000)
        vol = rng.integers(200, 3000, n)
        side = rng.choice(['B', 'S'], n, p=[0.55, 0.45])
    secs = np.sort(rng.integers(9 * 3600 + 30 * 60, 15 * 3600, n))
    drift = 0.0008 if archetype == '游资' else 0.0
    price = base_price * (1 + np.cumsum(rng.normal(drift / n, 0.0006, n)))
    price = np.round(price, 2)
    return pd.DataFrame({'sec': secs, 'BS标志': side,
                         '成交价格': price, '成交数量': vol})


def _gen_orders(rng, archetype, trades):
    """按原型生成逐笔委托（SSE：A新增/D撤单）。量化撤单最多。"""
    n = len(trades)
    if archetype == '量化':
        n_extra, cancel_p = int(n * 1.5), 0.45
    elif archetype == '游资':
        n_extra, cancel_p = int(n * 0.3), 0.08
    else:
        n_extra, cancel_p = int(n * 0.4), 0.12
    m = n + n_extra
    secs = np.sort(rng.integers(9 * 3600 + 30 * 60, 15 * 3600, m))
    otype = rng.choice([ORDER_ADD, ORDER_CANCEL], m, p=[1 - cancel_p, cancel_p])
    side = rng.choice(['B', 'S'], m)
    price = np.round(trades['成交价格'].mean() * (1 + rng.normal(0, 0.003, m)), 2)
    vol = rng.integers(100, 5000, m)
    return pd.DataFrame({'sec': secs, '委托类型': otype, '委托代码': side,
                         '委托价格': price, '委托数量': vol})


def _gen_snapshot(rng, archetype, trades, code, date_str):
    """3 秒快照 + 十档盘口，累计量/额。"""
    secs = np.arange(9 * 3600 + 30 * 60, 15 * 3600, 3)
    n = len(secs)
    base = trades['成交价格'].iloc[0]
    trend = 0.02 if archetype == '游资' else 0.0
    price = np.round(base * (1 + trend * np.linspace(0, 1, n) +
                             np.cumsum(rng.normal(0, 0.0004, n))), 2)
    tick_vol = rng.integers(500, 5000, n)
    cum_vol = np.cumsum(tick_vol)
    cum_amt = np.cumsum(tick_vol * price)
    d = {
        '万得代码': code, '交易所代码': 'SH', '自然日': int(date_str),
        '时间': _sec_to_hhmmssmmm(secs), '成交价': price, '成交量': tick_vol,
        '成交额': np.round(tick_vol * price, 2), '成交笔数': rng.integers(1, 20, n),
        'BS标志': rng.choice(['B', 'S'], n), '当日累计成交量': cum_vol,
        '当日成交额': np.round(cum_amt, 2), '最高价': np.maximum.accumulate(price),
        '最低价': np.minimum.accumulate(price), '开盘价': base,
        '前收盘': round(base * 0.99, 2),
        '加权平均叫卖价': np.round(price + 0.02, 2),
        '加权平均叫买价': np.round(price - 0.02, 2),
        '叫卖总量': rng.integers(50000, 200000, n),
        '叫买总量': rng.integers(50000, 200000, n),
    }
    imb = 1.5 if archetype == '游资' else 1.0
    for lv in range(1, 11):
        d[f'申卖价{lv}'] = np.round(price + 0.01 * lv, 2)
        d[f'申卖量{lv}'] = rng.integers(1000, 8000, n)
        d[f'申买价{lv}'] = np.round(price - 0.01 * lv, 2)
        d[f'申买量{lv}'] = (rng.integers(1000, 8000, n) * imb).astype(int)
    return pd.DataFrame(d)


def _write_gbk(df, path):
    df.to_csv(path, index=False, encoding='gbk')


def generate(base_dir, date_str='20260618', n_per_type=3):
    """生成 len(ARCHETYPES)*n_per_type 只股票的三源 CSV，返回股票代码列表。"""
    rng = np.random.default_rng(RANDOM_SEED)
    codes = []
    idx = 600000
    for archetype in ARCHETYPES:
        for _ in range(n_per_type):
            code = str(idx)
            idx += 1
            codes.append(code)
            out = os.path.join(base_dir, date_str, code)
            os.makedirs(out, exist_ok=True)
            base_price = round(rng.uniform(8, 40), 2)
            trd = _gen_trades(rng, archetype, base_price)
            ordf = _gen_orders(rng, archetype, trd)
            hq = _gen_snapshot(rng, archetype, trd, code, date_str)
            trd_out = _scale_price(trd, ['成交价格'])   # 价格 × PRICE_SCALE，对齐真实数据
            trd_out['时间'] = _sec_to_hhmmssmmm(trd_out['sec'])
            _write_gbk(trd_out[['时间', 'BS标志', '成交价格', '成交数量']],
                       os.path.join(out, '逐笔成交.csv'))
            ord_out = _scale_price(ordf, ['委托价格'])
            ord_out['时间'] = _sec_to_hhmmssmmm(ord_out['sec'])
            _write_gbk(ord_out[['时间', '委托类型', '委托代码', '委托价格', '委托数量']],
                       os.path.join(out, '逐笔委托.csv'))
            _write_gbk(_scale_price(hq, _PRICE_COLS_CN), os.path.join(out, '行情.csv'))
    return codes


if __name__ == '__main__':
    n = generate('./data')
    print(f'已生成 {len(n)} 只股票合成数据到 ./data/20260618/：{n}')
