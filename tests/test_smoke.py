import subprocess, sys, os
import pandas as pd
from tools.make_sample_data import generate
from src.config import CAPITAL_TYPES, INTENTIONS, PATTERN_NAMES
from main import run

def test_pipeline_end_to_end(tmp_path):
    generate(str(tmp_path), '20260618', n_per_type=5)   # 15 只
    out = tmp_path / 'out'
    run(data_dir=str(tmp_path), dates=['20260618'], out_dir=str(out), sample=None, limit=None)
    pat = pd.read_csv(out / 'pattern_reco.csv', dtype=str)
    res = pd.read_csv(out / 'predict_result.csv', dtype=str)
    assert len(pat) == 15 and len(res) == 15
    assert res['capital_type'].isin(CAPITAL_TYPES).all()
    assert res['capital_intention'].isin(INTENTIONS).all()
    assert pat['pattern_type'].isin(PATTERN_NAMES).all()
    assert res['capital_type'].nunique() >= 2
