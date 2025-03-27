# Multi Template MCMC贝叶斯分析

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

多模板MCMC贝叶斯分析是一个用于能谱拟合的Python工具包，它利用马尔可夫链蒙特卡洛（MCMC）方法进行贝叶斯分析，能够同时拟合多个模板谱。

[English README](README_EN.md)

## 特性

- 支持多模板能谱的同时拟合
- 基于emcee的高效MCMC采样
- 多种先验分布选择（均匀分布、正态分布、对数正态分布、截断正态分布）
- 全面的后验分布与拟合结果分析
- 生成美观的可视化图表
- 支持中英文双语HTML报告生成
- 交互式图表支持（使用Plotly）
- 自动时间戳输出目录
- 详细的参数统计分析（均值、中位数、方差、多种置信区间）

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/username/mtmcmc.git
cd mtmcmc
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

1. 编辑`config.py`文件，设置数据路径、MCMC参数和其他选项
2. 运行主程序：

```bash
python mtmcmc.py
```

### 使用示例脚本

项目包含一个示例脚本，可以生成合成数据并运行分析：

```bash
python example.py
```

## 配置参数

编辑`config.py`文件可以配置以下参数：

### 数据路径配置
- `TARGET_SPECTRUM`: 目标能谱数据文件路径
- `TEMPLATE_SPECTRA`: 模板能谱数据文件路径列表

### 输出配置
- `OUTPUT_DIR`: 结果输出目录
- `ADD_TIMESTAMP`: 是否在输出目录中添加时间戳子目录

### MCMC参数配置

在`config.py`中，您可以配置以下MCMC相关参数：

```python
# MCMC walkers数量
NWALKERS = 32

# MCMC采样步数
NSTEPS = 5000

# MCMC预热步数
BURNIN = 1000

# 是否显示进度条
PROGRESS = True

# 使用的CPU核心数，None表示使用全部可用核心
NCORES = None

# MCMC移动策略配置
# 每个元素是一个元组 (move, weight)，weight表示该移动策略的使用权重
# 可用的移动策略:
# - emcee.moves.DESnookerMove(): 差分进化Snooker移动
# - emcee.moves.DEMove(): 差分进化移动
# - emcee.moves.GaussianMove(): 高斯移动
# - emcee.moves.KDEMove(): 核密度估计移动
# - emcee.moves.StretchMove(): 伸展移动
MCMC_MOVES = [
    (emcee.moves.DESnookerMove(), 0.8),  # 使用80%的Snooker移动
    (emcee.moves.DEMove(), 0.2),          # 使用20%的差分进化移动
]
```

MCMC移动策略（moves）是控制采样器如何探索参数空间的重要设置。程序提供了多种移动策略供选择：

1. **DESnookerMove**: 差分进化Snooker移动
   - 优点：对高维参数空间有很好的探索能力
   - 适用：复杂的高维参数空间

2. **DEMove**: 差分进化移动
   - 优点：结合了多个walkers的信息
   - 适用：需要walkers之间协作的采样

3. **GaussianMove**: 高斯移动
   - 优点：简单且高效
   - 适用：参数空间较为简单的情况

4. **KDEMove**: 核密度估计移动
   - 优点：能够适应复杂的后验分布
   - 适用：多峰分布或非高斯分布

5. **StretchMove**: 伸展移动
   - 优点：计算开销小
   - 适用：需要快速采样的情况

您可以通过调整`MCMC_MOVES`列表来配置不同的移动策略组合。每个移动策略都有一个权重值，表示该策略被使用的概率。权重值应该满足：

```python
sum(weight for _, weight in MCMC_MOVES) == 1.0
```

默认配置使用80%的Snooker移动和20%的差分进化移动，这种组合在大多数情况下都能提供良好的采样效果。您可以根据具体问题调整这些权重或添加其他移动策略。

### 先验分布配置
- `PRIORS`: 各模板的先验分布设置，支持以下分布类型：
  - `uniform`: 均匀分布，参数为min和max
  - `normal`: 正态分布，参数为mu和sigma
  - `lognormal`: 对数正态分布，参数为mu和sigma
  - `truncnorm`: 截断正态分布，参数为min、max、mu和sigma
- `DEFAULT_PRIOR`: 默认先验分布设置

### 误差处理和HTML报告选项
- `ERROR_HANDLING`: 误差处理方式（'target'、'template'或'both'）
- `HTML_REPORT`: 是否生成HTML报告
- `INTERACTIVE_PLOTS`: HTML报告中是否包含交互式图表
- `TEMPLATE_DIR`: HTML模板目录
- `HTML_LANGUAGES`: HTML报告语言设置，可选值: ["zh"], ["en"], ["zh", "en"]

## 模块结构

- `mtmcmc.py`: 主程序
- `data_loader.py`: 数据加载与预处理模块
- `model.py`: 模型定义和先验分布模块
- `mcmc_sampler.py`: MCMC采样模块
- `analyzer.py`: 结果分析模块
- `visualizer.py`: 可视化模块（支持中英文双语图表）
- `html_reporter.py`: HTML报告生成模块（支持中英文双语报告）
- `config.py`: 配置文件
- `example.py`: 示例脚本

## 数据格式

输入数据文件应为文本格式，每行三列：能量、计数、误差。例如：

```
0.0 10.5 1.2
0.1 11.2 1.3
...
```

## 结果分析

分析结果将保存在输出目录中，包括：

- 模板权重的后验分布
- 拟合结果与残差分析
- 模板贡献分析
- 误差贡献分析
- 参数统计分析（均值、中位数、方差、多种置信区间）
- 综合HTML报告（中文和/或英文）

### 参数统计分析

对每个拟合参数提供以下统计量：
- 中位数值
- 平均值
- 标准差
- 方差
- MAP估计值（最大后验概率估计）
- 68% 置信区间
- 95% 置信区间
- 99.7% 置信区间
- 上下误差范围

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 贡献

欢迎贡献代码、报告问题或提出改进建议。 