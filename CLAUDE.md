# AFAC2026 赛题一：市场参与者交易行为识别与资金流向分析

## 提交规范

### 时间规则
- 每天只能提交**前一个交易日**的结果
- 交易日当天 23:59 前提交，过期该日计 0 分
- A榜每天最多 3 次提交，取最后一次

### 文件格式
打包为 `submit.zip`，内含两个 CSV 文件（UTF-8 编码，LF 换行）：

#### pattern_reco.csv — 交易模式识别
```
stock_code,transaction_date,pattern_type,pattern_explanation
603316,20260630,大单吸筹,资金大笔挂单买入短时间内集中扫货
```
- `stock_code`：纯数字，不带 `.SH` 后缀
- `transaction_date`：前一个交易日，格式 YYYYMMDD
- `pattern_type`：交易模式类别名称
- `pattern_explanation`：模式说明文字
- 100 行（对应 100 只股票）

#### predict_result.csv — 资金类型与意图识别
```
stock_code,transaction_date,capital_type,capital_intention
603316,20260630,游资,买入
```
- `stock_code`：同上
- `transaction_date`：同上
- `capital_type`：`游资` / `量化` / `散户` 三选一
- `capital_intention`：`买入` / `卖出` / `T0交易` 三选一
- 100 行（对应 100 只股票）

### 股票列表
来源于 `stock_sample.csv`（由官方 `股票样本.xlsx` 转存，列 `stock_code,stock_name`），共 100 只沪市股票，提交时去掉 `.SH` 后缀。

### 评分
- 总分 = 交易模式识别分 × 0.4 + 参与者识别分 × 0.6
- Task1 评估：类间区分度 + 类内聚合度（Wasserstein + DTW 距离）
- Task2 评估：加权 F1 Score

#### A榜评分实测结论（2026-07-07，平台提交对照实验）
- **Task1 只看聚类划分**：`pattern_type` 标签名、`pattern_explanation` 均**不计分**——标签双射改名（大单吸筹→组1…）、说明置换后分数不变。
- **Task2 只有 `capital_type` 计分**，为**监督加权 F1、标签身份重要**——三类循环轮换（游资→量化→散户→游资）后分数改变。
- **`capital_intention` 不计分**——随机打乱后分数不变。
- 推论：A榜只需优化 ①`pattern_type` 的聚类划分质量、②`capital_type` 的三分类准确率；explanation 与 intention 是 B 榜可解释性。

## 虚拟环境
```bash
source .venv/bin/activate
```
