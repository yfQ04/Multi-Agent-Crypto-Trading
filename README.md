# Multi-Agent Synergistic Framework for Cryptocurrency Trading

> 一个将市场情绪、技术指标、多个 LLM 决策角色与回测评估串联起来的 BTC 研究项目。

**项目结果速览** · 回测区间：`2025-03-10` — `2026-03-10` · 标的：**BTC**

| 多 Agent 策略累计收益 | 年化收益 | Sharpe Ratio | 最大回撤 |
|:---:|:---:|:---:|:---:|
| **+48.09%** | **+47.93%** | **1.34** | **−20.78%** |

## 项目简介

本项目实现一个面向 BTC 的多 Agent 日频研究与回测流程。系统先整理新闻/社交媒体情绪和价格技术指标，再让多种角色分别分析、辩论、控制风险并形成最终执行建议。回测报告将多 Agent 策略与 Buy & Hold、SMA、布林带和单 LLM 基线进行比较。

仓库已包含历史价格、情绪数据、Agent 日志以及生成好的图表和指标文件。查看下方结果不需要配置 API；配置密钥后可以运行模型流程并重新生成结果。

## 回测结果

下表来自仓库中的 [strategy_metrics_summary.csv](Multi-agent%20Crypto%20Trading/visualization/strategy_visualization_2025-03-10_2026-03-10_default_now/strategy_metrics_summary.csv)。收益和风险指标均为该历史区间的回测结果，不代表未来表现。

| 策略 | 累计收益 | 年化收益 | 年化波动率 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| **Multi-Agent** | **48.09%** | **47.93%** | 35.74% | **1.34** | −20.78% |
| SingleLLM | 15.91% | 15.86% | 25.16% | 0.63 | −25.45% |
| SingleBollinger (10, 2.5) | 5.99% | 5.98% | **12.82%** | 0.47 | **−7.64%** |
| SMA (7/20) | −5.26% | −5.25% | 24.02% | −0.22 | −36.08% |
| Buy & Hold | −12.20% | −12.17% | 44.64% | −0.27 | −49.94% |

### 累计收益对比

![2025-03-10 至 2026-03-10 各策略累计收益对比](https://raw.githubusercontent.com/yfQ04/Multi-Agent-Crypto-Trading/main/Multi-agent%20Crypto%20Trading/visualization/strategy_visualization_2025-03-10_2026-03-10_default_now/01_cumulative_return_curve.png)

### 回撤对比

![各策略回撤曲线](https://raw.githubusercontent.com/yfQ04/Multi-Agent-Crypto-Trading/main/Multi-agent%20Crypto%20Trading/visualization/strategy_visualization_2025-03-10_2026-03-10_default_now/02_drawdown_curve.png)

### 交易与执行概览

![多 Agent 策略交易执行概览](https://raw.githubusercontent.com/yfQ04/Multi-Agent-Crypto-Trading/main/Multi-agent%20Crypto%20Trading/visualization/strategy_visualization_2025-03-10_2026-03-10_default_now/13_trade_execution_overview.png)

> **结果说明：** 指标取自项目已保存的回测输出；年化指标按日频、每年 365 天计算，Sharpe 的无风险利率设为 0。实际交易中的手续费、滑点、流动性、数据时点和模型输出都可能显著影响结果。此项目用于研究和演示，不构成投资建议。

## Agent 工作流

```mermaid
flowchart LR
    D[历史价格 / 新闻 / 社交数据] --> A1[Agent 1<br/>情绪分析]
    D --> A2[Agent 2<br/>技术指标]
    A1 --> A34[Agent 3 & 4<br/>多空分析]
    A2 --> A34
    A34 --> A5[Agent 5<br/>交易方案]
    A5 --> A6[Agent 6<br/>风险评估]
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
未选择任何文件
Attach files by dragging & dropping, selecting or pasting them.
Content preview
Workbench
Press Delete to close the focussed tab.

README.md


180 lines · 8 KB

README.md file contents
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
# Multi-Agent Synergistic Framework for Cryptocurrency Trading

> 一个将市场情绪、技术指标、多个 LLM 决策角色与回测评估串联起来的 BTC 研究项目。

**项目结果速览** · 回测区间：`2025-03-10` — `2026-03-10` · 标的：**BTC**

| 多 Agent 策略累计收益 | 年化收益 | Sharpe Ratio | 最大回撤 |
|:---:|:---:|:---:|:---:|
| **+48.09%** | **+47.93%** | **1.34** | **−20.78%** |

## 项目简介

本项目实现一个面向 BTC 的多 Agent 日频研究与回测流程。系统先整理新闻/社交媒体情绪和价格技术指标，再让多种角色分别分析、辩论、控制风险并形成执行决策，最后统一进行策略回测与可视化评估。

仓库已包含历史价格、情绪数据、Agent 日志以及生成好的图表和指标文件。查看下方结果不需要配置 API；配置密钥后可以运行模型流程并重新生成结果。

## 回测结果

下表来自仓库中的 [strategy_metrics_summary.csv](./Multi-agent%20Crypto%20Trading/visualization/strategy_visualization_2025-03-10_2026-03-10_default_now/strategy_metrics_summary.csv)。收益和风险指标均基于同一区间计算。

| 策略 | 累计收益 | 年化收益 | 年化波动率 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| **Multi-Agent** | **48.09%** | **47.93%** | 35.74% | **1.34** | −20.78% |
| SingleLLM | 15.91% | 15.86% | 25.16% | 0.63 | −25.45% |
| SingleBollinger (10, 2.5) | 5.99% | 5.98% | **12.82%** | 0.47 | **−7.64%** |
| SMA (7/20) | −5.26% | −5.25% | 24.02% | −0.22 | −36.08% |
| Buy & Hold | −12.20% | −12.17% | 44.64% | −0.27 | −49.94% |

### 累计收益对比

![2025-03-10 至 2026-03-10 各策略累计收益对比](./Multi-agent%20Crypto%20Trading/visualization/strategy_visualization_2025-03-10_2026-03-10_default_now/01_cumulative_return_curve.png)

### 回撤对比

![各策略回撤曲线](./Multi-agent%20Crypto%20Trading/visualization/strategy_visualization_2025-03-10_2026-03-10_default_now/02_drawdown_curve.png)

### 交易与执行概览

![多 Agent 策略交易执行概览](./Multi-agent%20Crypto%20Trading/visualization/strategy_visualization_2025-03-10_2026-03-10_default_now/03_trade_execution_overview.png)

> **结果说明：** 指标取自项目已保存的回测输出；年化指标按日频、每年 365 天计算，Sharpe 的无风险利率设为 0。实际交易中的手续费、滑点、流动性约束及交易规则可能影响实盘表现。

## Agent 工作流

```mermaid
flowchart LR
    D[历史价格 / 新闻 / 社交数据] --> A1[Agent 1<br/>情绪分析]
    D --> A2[Agent 2<br/>技术指标]
    A1 --> A34[Agent 3 & 4<br/>多空分析]
    A2 --> A34
    A34 --> A5[Agent 5<br/>交易方案]
    A5 --> A6[Agent 6<br/>风险评估]
    A6 --> A7[Agent 7<br/>最终决策]
    A7 --> A8[Agent 8<br/>执行审批]
    A1 -. 独立基线 .-> A9[Agent 9<br/>SingleLLM]
    A2 -. 回测输入 .-> BT[策略回测与报告]
    A8 --> BT
    A9 --> BT
```

| 阶段 | 职责 |
|---|---|
| Agent 1 | 新闻及社交文本情绪处理 |
| Agent 2 | 价格数据和 RSI、MACD、均线等技术信息 |
| Agent 3–4 | 分别生成看多与看空论据 |
| Agent 5 | 综合观点，形成交易方案 |
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
