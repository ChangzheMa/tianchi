import glob, shutil, zipfile
import pandas as pd
from tools.make_sample_data import generate
from src.config import CAPITAL_TYPES, INTENTIONS, PATTERN_NAMES
from main import run

def test_pipeline_end_to_end(tmp_path):
    generate(str(tmp_path), '20260618', n_per_type=5)   # 15 只
    out = tmp_path / 'out'
    run(data_dir=str(tmp_path), dates=['20260618'], out_dir=str(out), sample=None, limit=None)

    # cleanup=True 后中间 CSV 已删、只剩 zip；从 zip 解压到临时子目录校验，校验完删除解压产物
    zips = glob.glob(str(out / 'submit_*.zip'))
    assert len(zips) == 1, f'期望唯一提交包，实得 {zips}'
    unzip_dir = tmp_path / 'unzip'
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(unzip_dir)
    try:
        pat = pd.read_csv(unzip_dir / 'pattern_reco.csv', dtype=str)
        res = pd.read_csv(unzip_dir / 'predict_result.csv', dtype=str)
        assert len(pat) == 15 and len(res) == 15
        assert res['capital_type'].isin(CAPITAL_TYPES).all()
        assert res['capital_intention'].isin(INTENTIONS).all()
        assert pat['pattern_type'].isin(PATTERN_NAMES).all()
        assert res['capital_type'].nunique() >= 2
    finally:
        shutil.rmtree(unzip_dir)                          # 删除解压产物
