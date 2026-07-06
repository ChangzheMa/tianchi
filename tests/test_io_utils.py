import pandas as pd
from src import io_utils

def test_save_results_legal(tmp_path):
    df_pat = pd.DataFrame({'stock_code': ['600000.SH', '000001.SZ'],
                           'transaction_date': ['20260618', '20260618'],
                           'pattern_type': ['大单吸筹', '日内套利'],
                           'pattern_explanation': ['x', 'y']})
    df_res = pd.DataFrame({'stock_code': ['600000.SH', '000001.SZ'],
                           'transaction_date': ['20260618', '20260618'],
                           'capital_type': ['游资', '散户'],
                           'capital_intention': ['买入', 'T0交易']})
    io_utils.save_results(df_pat, df_res, str(tmp_path))
    p = pd.read_csv(tmp_path / 'pattern_reco.csv', dtype=str)
    assert list(p['stock_code']) == ['600000', '000001']
    raw = open(tmp_path / 'predict_result.csv', 'rb').read()
    assert not raw.startswith(b'\xef\xbb\xbf')
    assert b'\r\n' not in raw

def test_illegal_capital_type_raises(tmp_path):
    import pytest
    df_pat = pd.DataFrame({'stock_code': ['600000'], 'transaction_date': ['20260618'],
                           'pattern_type': ['大单吸筹'], 'pattern_explanation': ['x']})
    df_res = pd.DataFrame({'stock_code': ['600000'], 'transaction_date': ['20260618'],
                           'capital_type': ['量化机构'], 'capital_intention': ['买入']})
    with pytest.raises(AssertionError):
        io_utils.save_results(df_pat, df_res, str(tmp_path))
