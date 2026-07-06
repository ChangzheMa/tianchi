# 04 · Baseline 方案详解

> 来源：`baseline教程.txt`。这是**面向真实 A/B 榜三源 CSV 数据**的新版方案设计。
> ⚠️ 根目录 `main.py` 是**旧版**（单 Excel + 二分类），与本文所述**不一致**，勿混淆，详见文档 06。

## 一、整体架构（5 层闭环）

```
三源 L2 CSV（行情/逐笔成交/逐笔委托，GBK 中文列名）
        ↓
数据预处理层：HQ_COL_MAP 中文→英文映射、_parse_time_to_seconds 时间标准化、
              列裁剪 + dtype 优化、异常值过滤、时序排序
        ↓
特征工程层：extract_features 提取 10 大类约 100 维特征，按「个股+交易日」聚合为特征矩阵
        ↓
建模推理层
    ├ Task1：RobustScaler+Rank 混合归一化 → 4 模型融合投票 → PATTERN_RULES 贪心匹配
    │          10 种模式 → 多样性校准(6%~18%)
    └ Task2：两阶段识别（KMeans 行为组 → 组内 Z-score → 游资/量化/散户三因子打分）
              → buy/sell_signal 多维意图判定 → 多样性校准(22%~45%)
        ↓
后处理层：_post_filter_calibrate 在目标股票子集上重新平衡分布
        ↓
结果输出层：save_results 格式校验 → pattern_reco.csv + predict_result.csv
```

## 二、数据预处理

三源独立读取（`load_snapshot / load_trades / load_orders`），`HQ_COL_MAP` 统一中文列名为英文。

```python
def _parse_time_to_seconds(time_val):
    """HHMMSSmmm → 日内秒数"""
    t = np.int64(time_val)
    return (t // 10000000) * 3600 + ((t // 100000) % 100) * 60 + ((t // 1000) % 100)
```

要点：
- 时间 `HHMMSSmmm` 整数 → 日内秒数后做时段判定（开盘 30 分钟、尾盘 10 分钟、8 个半小时时段）。
- **逐笔成交.csv 直接分级**：`amount = 成交价格 × 成交数量`，无需对累计值 diff。
- **BS 标志**直接给主动买卖方向；**委托类型**（U/1/0）是撤单特征核心。

## 三、特征工程（10 大类约 100 维）

| 类别 | 维度 | 说明 |
|------|------|------|
| OSS 大单分级 | 8 | 按**成交额**分级：超大≥50万 / 大≥10万 / 中≥4万 / 小<4万，金额与笔数占比 |
| TRD 交易结构 | 4 | 平均成交额/量、成交额标准差、总笔数 |
| RS 订单时序 | 6 | 间隔变异系数、拆单相似度、爆发率(<1s)、高频占比(<0.1s)、间隔偏度/峰度 |
| CB 撤单行为 | 7 | 基于委托类型=='U'：撤单率、成交委托比、撤单金额占比、买/卖撤单分化 |
| AP 主动成交 | 8 | 基于 BS 标志：主动买/卖占比、净成交占比、单边强度、连续买卖笔数 |
| PI 日内时段 | 10 | 8 个半小时时段占比、开盘 30min/尾盘 10min 占比、时段集中度 |
| PD 价格发现 | — | 开盘跳空、日内收益、振幅、距涨停、VWAP 偏离、价格反转、大单方向一致性 |
| OBP 盘口衍生 | 12 | 十档价量：最优价差、盘口失衡度、大单挂单占比、加权价差统计、深度变化率 |
| 盘口动态 | 9 | 价差波动、相对价差、买卖深度比变化、净买盘变化 |
| **资金集中度** | 8 | 赫芬达尔指数、Top5%大单占比、成交额熵、委托成交比、撤单聚集度、间隔自相关、买卖不对称性、连续同向笔数 |

> ⚠️ OSS 分级口径：baseline 教程用**成交额金额阈值**（≥50万等）；旧版 main.py 用**成交量阈值**（≥50000 等）。两者不可混用。

## 四、Task1 — 4 模型融合聚类

**Step 1 归一化**：`_normalize_features_robust` = 50% RobustScaler + 50% Rank 百分位，对异常值鲁棒、量纲无关。

**Step 2 四模型投票**（`task1_multi_model_fusion`，k=10）：
- KMeans++（球形）
- DBSCAN（密度，自适应 eps = k-distance 90 分位，噪声点 -1 回退 KMeans）
- Agglomerative（层次）
- GMM（高斯混合，软分配）
- 多数投票融合，平票回退 KMeans。

**Step 3 语义映射**：`PATTERN_RULES`（10 种模式 × 6 特征 × 方向 +1/-1/0）贪心匹配——
按聚类大小排序，每簇选"得分最高且未被占用"的模式。

10 种模式：**大单吸筹 / 压单吸货 / 尾盘突袭 / 集合竞价异动 / 对倒拉升 / 盘中诱多 /
分时脉冲 / 涨停板打开 / 连续小单推升 / 日内套利**。

**Step 4 多样性校准**（10 轮退火）：每种模式占比约束在 **6%~18%**；
超比例模式把"次优得分最高"的样本迁出，欠比例模式把"目标得分提升最大"的样本迁入。

## 五、Task2 — 两阶段识别

**Stage 1 行为组聚类**：用 KMeans 在 10 个**与价格量纲无关**的行为特征上，
将全市场聚为 3~6 个行为组（`n_clusters = min(6, max(3, n//100))`），
解决大盘股/小盘股不可比问题（`trd_avg_trade_amount` 先 log1p）。

**Stage 2 组内归一化 + 三因子打分**：全局 min-max → **组内 Z-score**，再算三因子加权分，取最高：
- **游资**（15 维，侧重大额/单边/集中度/冲击）：`oss_mega_amount_pct×0.12 + pd_big_order_amount_pct×0.10 + ap_unilateral_intensity×0.10 + fd_herfindahl×0.09 + ...`
- **量化**（15 维，侧重规律性/高频/撤单/小单）：`rs_split_similarity×0.12 + cb_cancel_ratio×0.10 + rs_hft_ratio×0.09 + fd_order_trade_ratio×0.08 + ...`
- **散户**（13 维，侧重小单/不规则/宽价差/低集中度）：`oss_small_amount_pct×0.15 + rs_interval_cv×0.11 + |spread|×0.09 + ...`

**Stage 3 资金类型校准**（5 轮）：游资/量化/散户占比约束在 **22%~45%**。

**意图判定**：`buy_signal / sell_signal` 各 7 维加权（权重 1.0/0.7/0.6/0.5/0.4/0.3/0.3，满 3.8），
覆盖净买卖比、买卖成交额比、VWAP 偏离、大单买入占比、盘口失衡快照/均值、日内收益。
```
bs≥0.9 且 ss<0.6 → 买入
ss≥0.9 且 bs<0.6 → 卖出
bs、ss 均弱(<0.5) → T0交易
bs>ss×1.5 → 买入 ；ss>bs×1.5 → 卖出 ；否则默认 T0交易
```
**意图校准**（5 轮）：买入/卖出约束 18%~40%，T0 交易 ≥12%。

## 六、后处理与输出

- `_post_filter_calibrate`：在**目标股票子集**（剔除建模辅助但非目标输出的股票）上做第二轮轻量校准，
  模式 5%~20%、资金合理三类分布，并同步更新 `pattern_explanation`。
- `save_results`：字段顺序 + 合法值校验（`capital_type ∈ {游资,量化,散户}`、
  `capital_intention ∈ {买入,卖出,T0交易}`），UTF-8-sig 输出。
  > 注：提交编码建议改为**不带 BOM 的 UTF-8 + LF**，与官方要求一致（详见文档 02）。

## 七、运行与环境

```bash
python main.py                                          # 默认跑 20260618 目录全部股票
python main.py -d 20260618 20260619 -b ./data -o ./out  # 指定日期/目录/输出
python main.py -n 20                                    # 仅跑前 20 只（调试）
```
- 环境：Python ≥3.8；`pandas, numpy, scikit-learn, openpyxl`
- 全局 `RANDOM_SEED = 42` 固定，确保可复现。

## 八、离线评估

- **Task1**：本地用 `silhouette_score / calinski_harabasz_score` 参考（需基于融合后的 final_labels 计算）；
  线上还含 Wasserstein、DTW。
- **Task2**：本地无真值，只能看分布是否合理（三类各 22%~45%）；线上用实盘回溯真值算加权 F1。
- **校准达标观察**：模式 6%~18%、资金 22%~45%、意图 T0≥12%；未达标调 `target_min/max` 或加迭代轮数。
