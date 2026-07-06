import os, pandas as pd
from tools.make_sample_data import generate

def test_generate_creates_three_source_csv(tmp_path):
    codes = generate(str(tmp_path), '20260618', n_per_type=1)
    assert len(codes) == 3                       # 三种原型各1只
    for code in codes:
        d = tmp_path / '20260618' / code
        for fn in ['行情.csv', '逐笔成交.csv', '逐笔委托.csv']:
            assert (d / fn).exists()
    df = pd.read_csv(tmp_path / '20260618' / codes[0] / '逐笔成交.csv', encoding='gbk')
    assert list(df.columns) == ['时间', 'BS标志', '成交价格', '成交数量']
    assert df['时间'].iloc[0] >= 92500000 and df['时间'].iloc[0] <= 150500000
    hq = pd.read_csv(tmp_path / '20260618' / codes[0] / '行情.csv', encoding='gbk')
    assert '申买价1' in hq.columns and '申卖量10' in hq.columns
