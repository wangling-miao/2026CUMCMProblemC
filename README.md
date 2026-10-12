# 微网购电与储能优化

2026 年全国大学生数学建模竞赛 C 题研究项目。论文《基于滚动时域优化与情景分析的微网购电储能模型》研究光伏、负载和电价变化下的购电计划、储能调度与合同调整。

[阅读论文](paper/main.pdf) · [模型源码](code/model/src/cumcm_v11) · [正式结果](results/official) · [实验统计](results/experiments/analysis)

## 方法

模型以 10 分钟为时间步长，用线性规划处理功率平衡、储能容量、充放电效率与购电约束；结合历史预测误差和滚动时域优化生成日前计划。在允许合同调整的场景下，使用组合预测和历史情景净增益筛选，在 6、12、18 时评估调整候选。费用按实际生效合同、紧急购电和实现电价结算。

问题二与四的不可调整场景使用最近 28 个小时档案观测形成保护量；问题三与四的可调整场景使用最近 5 个完成日及四个候选。正式可调整方案采用历史路径净增益均值大于 5 元的门槛，默认 `screen_worst_case=False`。

## 参考结果

年度仿真从 2025 年 1 月 1 日、初始储电量 6000 kWh 开始。1 月用于形成预测历史和储能状态，评价区间为 2 月 1 日至 12 月 31 日，共 334 天。问题一为单独的典型日模型。

| 问题 | 场景 | 购电费用 / 元 | 结果文件 |
|---|---|---:|---|
| 一 | 典型日确定性优化 | 35,126.95 | [result1.xlsx](results/official/result1.xlsx) |
| 二 | 固定电价，不可调整 | 13,790,848.69 | [result2.xlsx](results/official/result2.xlsx) |
| 三 | 固定电价，可调整 | 13,298,484.49 | [result3.xlsx](results/official/result3.xlsx) |
| 四-2 | 波动电价，不可调整 | 14,526,595.92 | [result4-2.xlsx](results/official/result4-2.xlsx) |
| 四-3 | 波动电价，可调整 | 14,018,361.30 | [result4-3.xlsx](results/official/result4-3.xlsx) |

问题三相对问题二节省 3.57%，问题四可调整场景相对不可调整场景节省 3.50%。这些结果对应给定历史数据和配置，不构成全局最优性证明。

## 快速开始

需要 Python 3.12 或更新版本。以下命令在仓库根目录执行；`requirements.txt` 固定数值计算依赖，便于控制环境差异。

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

先运行典型日问题：

```bash
python scripts/run_model.py --confirm-recompute --only q1
```

短回放可用于检查年度模型入口；两天输出不代表 334 天评价结果：

```bash
python scripts/run_model.py --confirm-recompute --only q3 --max-days 2 --jobs 1
```

完整重算：

```bash
python scripts/run_model.py --confirm-recompute --only all --jobs 4
```

重算写入 `results/recomputed/`。参考工作簿位于 `results/official/`，原始附件和结果模板位于 `data/raw/data/`。模型包保留名称 `cumcm_v11`，运行入口直接加载仓库内源码，无需安装该包。年度任务涉及大量线性规划求解，耗时取决于硬件和并行进程数。

发布检查使用 Python 3.12.13 和上述固定依赖：34 项模型测试、36 项实验运行器检查、问题一费用回归及问题三两天回放均通过。本次发布未重新执行完整年度任务和 66 组实验矩阵。

## 论文编译

`paper/main.pdf` 是仓库版正文，LaTeX 入口为 `paper/main.tex`，参考文献位于 `paper/ref.bib`。需安装 TeX Live（XeLaTeX、Biber），并提供模板所用字体：Times New Roman、DejaVu Sans、DejaVu Sans Mono、宋体、黑体与楷体。字体设置位于 `paper/format.cls`。

```bash
python scripts/build_paper.py
```

## 实验与解释范围

[实验说明](experiments/README.md) 记录正式方案与历史实验对照的区别。仓库保留 66 组回顾性消融和敏感性实验的汇总统计、分月配对结果及风险诊断，未收录全部逐次求解日志。历史实验的可调整对照启用了最坏情景损失限制，不能将其直接视为当前正式方案。

本数据参与过配置比较。移动块重采样区间描述同一段历史中的配对差稳定性，不是独立样本外验证或未来全年费用区间；敏感性曲线中的最低点也不自动替换正式参数。

## 目录

| 路径 | 内容 |
|---|---|
| `paper/` | 论文 PDF、LaTeX 正文、参考文献、表格和矢量图 |
| `code/model/src/cumcm_v11/` | 数据读取、预测、优化、情景筛选、仿真与结算 |
| `code/model/tests/` | 结算、物理约束、因果性及回归检查 |
| `data/raw/data/` | 竞赛输入附件及输出模板 |
| `results/official/` | 五份正式参考工作簿 |
| `results/diagnostics/` | 年度参考轨迹、配置与约束检查 |
| `results/experiments/analysis/` | 历史实验统计及风险诊断 |
| `experiments/` | 实验矩阵、运行器与检查程序 |
| `scripts/` | 模型重算及论文编译入口 |
