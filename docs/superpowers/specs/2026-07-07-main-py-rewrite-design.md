# main.py 重写设计（精简忠实版）

- 日期：2026-07-07
- 目标文件：`main.py` + `src/` 包 + `tools/make_sample_data.py`
- 依据：`doc/04-Baseline方案详解.md`（baseline 教程新版）、`doc/06-关键风险与行动清单.md`

## 1. 背景与目标

现有根目录 `main.py` 是**旧版样例**：`capital_type` 用 `{游资, 量化机构}` 二分类、
数据层假设单个 Excel（`symbol`/UTC 毫秒/`bids-asks` JSON），与真实 A/B 榜的
「日期/股票/三源 GBK 中文列名 CSV」结构和三分类要求**完全不符**，无法运行于真实数据。

本设计将 `main.py` 重写为**符合 baseline 教程新版**的实现：三源 CSV 输入、约 100 维特征、
Task1 无监督聚类 + 模式映射、Task2 两阶段三分类资金识别、多样性校准、合法格式输出。

**成功标准**：
1. 结构符合真实数据（三源 CSV / GBK / HHMMSSmmm / 三分类）。
2. 在合成数据上能端到端跑通，产出两个格式合法的 CSV。
3. 模块化、单一职责、可独立测试。

## 2. 范围

**做（In scope）**：
- 三源 CSV 数据加载与预处理。
- 10 大类约 100 维特征工程（忠实教程）。
- Task1：KMeans(k=10) 聚类 + `PATTERN_RULES` 贪心映射 10 种模式 + 分布校准(6%~18%)。
- Task2：行为组聚类 → 组内 Z-score → 游资/量化/散户三因子打分 + 分布校准(22%~45%)；简版意图判定。
- 合成数据生成脚本 + 冒烟测试。
- 结果格式校验与合法输出（UTF-8 无 BOM + LF，去代码后缀）。

**不做（Out of scope，留待有数据后迭代）**：
- 4 模型融合（DBSCAN/Agglomerative/GMM 投票）——仅留扩展点，默认单 KMeans。
- 7 维意图信号 + 意图分布校准（A 榜不计分）。
- 后处理二次校准 `_post_filter_calibrate`。
- Wasserstein/DTW 本地评估（线上指标，不引 scipy）。
- XGBoost/LLM/时序模型等提分方案（见 `doc/05`）。

## 3. 目录结构

```
E:/src/tianchi/
├── main.py                 # 薄入口：argparse → 编排流水线
├── src/
│   ├── __init__.py
│   ├── config.py           # 常量与规则表（无逻辑）
│   ├── data_io.py          # 三源 CSV 读取 + 列名映射 + 时间解析
│   ├── features.py         # 特征提取 + 特征矩阵构建
│   ├── task1_pattern.py    # 聚类 + 模式映射
│   ├── task2_capital.py    # 两阶段资金识别 + 简版意图
│   ├── calibrate.py        # 通用分布校准
│   └── io_utils.py         # 股票样本读取、结果校验与保存
├── tools/
│   └── make_sample_data.py # 合成三源 CSV 测试数据
├── data/                   # 数据根（运行期）：<YYYYMMDD>/<股票代码>/三源CSV
└── 股票样本.xlsx
```

## 4. 各模块职责与接口

### 4.1 `src/config.py`（纯常量）
- `RANDOM_SEED = 42`
- `HQ_COL_MAP`：行情.csv 中文列名 → 英文（含申买/卖价1~10、量1~10）。
- OSS 金额阈值：超大 ≥500000 / 大 ≥100000 / 中 ≥40000 / 小 <40000（**按成交额**）。
- 交易时段：有效窗口 9:25–15:05；开盘 30min、尾盘 10min、8 个半小时时段边界（日内秒）。
- `N_PATTERNS = 10`；`PATTERN_RULES`：10 种模式 ×（名称, 解释, [(特征, 方向±1/0)]）。
  - 模式集：大单吸筹/压单吸货/尾盘突袭/集合竞价异动/对倒拉升/盘中诱多/分时脉冲/涨停板打开/连续小单推升/日内套利。
- 校准区间：模式 `(0.06, 0.18)`、资金 `(0.22, 0.45)`。
- 三因子特征权重：`SCORE_WEIGHTS = {'游资':[...], '量化':[...], '散户':[...]}`（特征名→权重）。

### 4.2 `src/data_io.py`
- `parse_time_to_seconds(time_val) -> int/ndarray`：HHMMSSmmm → 日内秒。
- `load_snapshot(stock, date, base) -> DataFrame`：读行情.csv（GBK, usecols+dtype），
  rename → 英文，加 `seconds`，按 seconds 排序。
- `load_trades(stock, date, base) -> DataFrame`：读逐笔成交.csv，`amount = 成交价格×成交数量`。
- `load_orders(stock, date, base) -> DataFrame`：读逐笔委托.csv（order_type/side/price/volume）。
- 约定：文件缺失时抛出**明确异常**（由上层捕获跳过该股）。

### 4.3 `src/features.py`
- `extract_features(stock, date, base) -> dict`：返回单只股票单日的特征字典（含 `stock_code`、
  `transaction_date`）。内部按类拆子函数：
  - OSS 大单分级（按成交额）、TRD 交易结构、RS 订单时序、CB 撤单行为（委托类型=='U'）、
    AP 主动成交（BS 标志）、PI 日内时段、PD 价格发现（行情快照）、OBP 盘口衍生（十档）、
    盘口动态、资金集中度（赫芬达尔/Top5%/委托成交比/间隔自相关等）。
  - 空/退化数据 → 该类特征填默认值。
- `build_feature_matrix(pairs, base) -> DataFrame`：对 (股票,日期) 列表逐个提特征，
  缺文件的跳过并告警；对数值列 `fillna(median)` → `fillna(0)` → 替换 inf。

### 4.4 `src/task1_pattern.py`
- `normalize_robust(df_feat) -> (ndarray, feat_cols)`：50% RobustScaler + 50% Rank 百分位混合。
- `cluster(feat_matrix, k) -> labels`：KMeans++（`n_init=10`, seed）。样本数<k 自动降 k。
  - 扩展点：预留 `ensemble=False` 参数，后续可加 GMM 投票。
- `map_clusters_to_patterns(df_feat, labels) -> DataFrame`：对每簇按 `PATTERN_RULES` 跨样本
  百分位方向打分，按簇大小排序贪心分配唯一模式 → `pattern_type` + `pattern_explanation`。

### 4.5 `src/task2_capital.py`
- `two_stage_capital(df_feat) -> DataFrame`：
  - Stage1：在 10 个量纲无关行为特征上 KMeans 聚 3~6 组（`trd_avg_trade_amount` 先 log1p）。
  - Stage2：全局 min-max → 组内 Z-score → 游资/量化/散户三因子加权得分 → argmax = `capital_type`。
- `judge_intention(df_feat) -> Series`：**简版**——净买卖比 + 盘口失衡 + VWAP 偏离，
  阈值判定 买入/卖出/T0交易（默认 T0）。

### 4.6 `src/calibrate.py`
- `calibrate_distribution(df, col, class_names, score_matrix, tmin, tmax, rounds) -> df`：
  通用迭代再平衡。超比例类把"次优得分差距最小"的样本迁出，欠比例类把"目标得分提升最大"的迁入。
  **模式校准与资金校准复用同一函数**（传不同 col/区间/得分矩阵）。

### 4.7 `src/io_utils.py`
- `load_stock_sample(path) -> set[str]`：读 `股票样本.xlsx`，代码去 `.SH/.SZ` 后缀。
- `save_results(df_pat, df_res, out_dir)`：
  - 断言字段名/顺序；`capital_type∈{游资,量化,散户}`、`capital_intention∈{买入,卖出,T0交易}`。
  - `stock_code` 去后缀为纯数字。
  - 写盘：`encoding='utf-8'`（**无 BOM**）、`lineterminator='\n'`（**LF**）、`index=False`。

### 4.8 `main.py`
- argparse：`--data ./data`、`--date/-d`（可多值，缺省=data/下全部日期）、`--out/-o ./out`、
  `--sample`（可选，过滤到 100 只目标股票）、`-n`（限量调试）。
- 编排：枚举 (股票,日期) → `build_feature_matrix` → `task1` + `task2` → `calibrate` → `save_results`。
  多日期合并输出。全程固定随机种子、相对路径。

## 5. 合成数据与验证

- `tools/make_sample_data.py`：生成 ~9 只股票 × 1 天，含 3 种行为原型：
  - **游资式**：大单集中、单边主动买入、时段集中、撤单少。
  - **量化式**：高频小单、拆单均匀、撤单频繁、多空均衡。
  - **散户式**：小单零散、间隔不规则、宽价差、无方向。
  - 输出为 GBK 中文列名三源 CSV，时间为 HHMMSSmmm，写入 `data/<日期>/<股票>/`。
- 冒烟测试：`python tools/make_sample_data.py && python main.py -d <日期>`
  → 断言产出两 CSV、值合法、三类分布非退化（非全同一类）。
- `tests/test_smoke.py`（极简 `pytest`）：默认交付——生成合成数据 → 跑完整流水线 →
  校验输出 schema、字段合法值、行数。低成本防回归；如不需要可在复核时否掉。

## 6. 错误处理

| 场景 | 处理 |
|------|------|
| 单股缺文件/空数据 | 跳过该股并告警，不中断整批 |
| 退化/NaN/Inf 特征 | 填默认值 / `nan_to_num` |
| 样本数 < 聚类 k | 自动降 k |
| 输出非法（字段/值） | 保存前 `assert` 炸掉，绝不产出非法提交 |

## 7. 约定（已与用户确认）

1. 数据目录：`data/<YYYYMMDD>/<股票代码>/{行情.csv,逐笔成交.csv,逐笔委托.csv}`，GBK，中文列名。
2. CLI：`python main.py` 跑全部日期；`-d` 指定日期；`-o` 输出目录；`--sample` 过滤 100 只；`-n` 限量。
3. 输出编码：UTF-8 **无 BOM** + LF（覆盖教程的 utf-8-sig）。
4. 依赖：`pandas / numpy / scikit-learn / openpyxl`（装进 `.venv`，暂不引 scipy）。
5. 旧 `main.py` 直接被覆盖（已在 git 历史中可追回）。

## 8. 交付物

- `main.py`、`src/`（7 个模块）、`tools/make_sample_data.py`、`tests/test_smoke.py`。
- 冒烟测试通过证据（生成数据 → 跑通 → 两个合法 CSV）。
- 提交（编码阶段完成后）。
