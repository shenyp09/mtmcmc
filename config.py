#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模板MCMC贝叶斯分析配置文件
"""

#############################################
# 数据路径配置
#############################################

# 目标能谱数据文件路径
TARGET_SPECTRUM = "data/synthetic_target.txt"

# 模板能谱数据文件路径列表
TEMPLATE_SPECTRA = [
    "data/synthetic_template1.txt",
    "data/synthetic_template2.txt",
    "data/synthetic_template3.txt",
]

#############################################
# 输出配置
#############################################

# 结果输出目录
OUTPUT_DIR = "results"

# 是否在输出目录中添加时间戳子目录
ADD_TIMESTAMP = True

#############################################
# MCMC参数配置
#############################################

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

#############################################
# 先验分布配置
#############################################

# 各模板的先验分布设置，每个模板可以设置不同的先验分布
PRIORS = [
    # 模板1的先验分布
    {"type": "uniform", "params": {"min": 0.0, "max": 10.0}},
    # 模板2的先验分布
    {"type": "normal", "params": {"mu": 1.0, "sigma": 0.5}},
    # 模板3的先验分布
    {"type": "lognormal", "params": {"mu": 0.0, "sigma": 0.5}},
]

# 如果PRIORS中没有为每个模板指定先验，则使用以下默认先验
DEFAULT_PRIOR = {"type": "uniform", "params": {"min": 0.0, "max": 10.0}}

#############################################
# 误差处理和HTML报告选项
#############################################

# 误差处理方式: 'target', 'template', 'both'
ERROR_HANDLING = "both"

# 是否生成HTML报告
HTML_REPORT = True

# HTML报告中是否包含交互式图表
INTERACTIVE_PLOTS = True

# HTML模板目录，None表示使用内置模板
TEMPLATE_DIR = None

# HTML报告语言设置，可选值: ["zh"], ["en"], ["zh", "en"]
HTML_LANGUAGES = ["zh", "en"]
