from tools.make_sample_data import generate
from src import data_io

def _setup(tmp_path):
    codes = generate(str(tmp_path), '20260618', n_per_type=1)
    return codes[0], '20260618', str(tmp_path)

def test_parse_time():
    assert data_io.parse_time_to_seconds(93000000) == 9 * 3600 + 30 * 60
    assert data_io.parse_time_to_seconds(145959000) == 14 * 3600 + 59 * 60 + 59

def test_load_snapshot(tmp_path):
    code, date, base = _setup(tmp_path)
    df = data_io.load_snapshot(code, date, base)
    assert 'price' in df.columns and 'seconds' in df.columns
    assert 'bid_price_1' in df.columns and 'ask_vol_10' in df.columns
    assert df['seconds'].is_monotonic_increasing

def test_load_trades(tmp_path):
    code, date, base = _setup(tmp_path)
    df = data_io.load_trades(code, date, base)
    assert set(['side', 'price', 'volume', 'amount', 'seconds']).issubset(df.columns)
    assert (df['amount'] == df['price'] * df['volume']).all()

def test_load_orders(tmp_path):
    code, date, base = _setup(tmp_path)
    df = data_io.load_orders(code, date, base)
    assert set(['order_type', 'side', 'seconds']).issubset(df.columns)
    assert df['order_type'].isin(['0', 'U', '1']).any()

def test_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        data_io.load_trades('999999', '20260618', str(tmp_path))
