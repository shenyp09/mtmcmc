# Multi Template MCMC贝叶斯分析

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

多模板MCMC贝叶斯分析是一个用于能谱拟合的Python工具包，它利用马尔可夫链蒙特卡洛（MCMC）方法进行贝叶斯分析，能够同时拟合多个模板谱。

## 特性

- 支持多模板能谱的同时拟合
- 基于emcee的高效MCMC采样
- 多种先验分布选择（均匀分布、正态分布、对数正态分布等）
- 全面的后验分布与拟合结果分析
- 生成美观的可视化图表
- 支持中英文双语HTML报告生成
- 交互式图表支持（使用Plotly）
- 自动时间戳输出目录

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/shenyp09/mtmcmc.git
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
- `NWALKERS`: MCMC walkers数量
- `NSTEPS`: MCMC采样步数
- `BURNIN`: MCMC预热步数
- `PROGRESS`: 是否显示进度条
- `NCORES`: 使用的CPU核心数，None表示使用全部可用核心

### 先验分布配置
- `PRIORS`: 各模板的先验分布设置
- `DEFAULT_PRIOR`: 默认先验分布设置

### 误差处理和HTML报告选项
- `ERROR_HANDLING`: 误差处理方式
- `HTML_REPORT`: 是否生成HTML报告
- `INTERACTIVE_PLOTS`: HTML报告中是否包含交互式图表
- `TEMPLATE_DIR`: HTML模板目录
- `HTML_LANGUAGES`: HTML报告语言设置，可选值: ["zh"], ["en"], ["zh", "en"]

## 模块结构

- `mtmcmc.py`: 主程序
- `data_loader.py`: 数据加载与预处理模块
- `model.py`: 模型定义模块
- `mcmc_sampler.py`: MCMC采样模块
- `analyzer.py`: 结果分析模块
- `visualizer.py`: 可视化模块
- `html_reporter.py`: HTML报告生成模块
- `config.py`: 配置文件
- `example.py`: 示例脚本

## 数据格式

输入数据文件应为文本格式，每行三列：能量、计数、误差。例如：

```
0.0 10.5 1.2
0.1 11.2 1.3
...
```

## 结果

分析结果将保存在输出目录中，包括：

- 模板权重的后验分布
- 拟合结果与残差分析
- 模板贡献分析
- 误差贡献分析
- 综合HTML报告

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 贡献

欢迎贡献代码、报告问题或提出改进建议。 