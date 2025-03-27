#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023-2024 MTMCMC Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

多模板MCMC贝叶斯分析配置文件
"""

import emcee

#############################################
# 数据路径配置
#############################################

# 目标能谱数据文件路径
TARGET_SPECTRUM = "data/410/hdat.txt"

# 模板能谱数据文件路径列表
TEMPLATE_SPECTRA = [
    "data/410/hsg.txt",
    "data/410/hb1.txt",
    "data/410/hb2.txt",
    "data/410/hb3.txt",
    "data/410/hb4.txt",
    # "data/410/hb5.txt",
    "data/410/hb6.txt",
    "data/410/hb7.txt",
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
    (emcee.moves.DEMove(), 0.2),  # 使用20%的差分进化移动
]

#############################################
# 先验分布配置
#############################################

# 先验分布类型:
# - uniform: 均匀分布，参数为min和max
#   例: {"type": "uniform", "params": {"min": 0.0, "max": 10.0}}
#
# - normal: 正态分布，参数为mu(均值)和sigma(标准差)
#   例: {"type": "normal", "params": {"mu": 1.0, "sigma": 0.5}}
#
# - lognormal: 对数正态分布，参数为mu(对数均值)和sigma(对数标准差)
#   例: {"type": "lognormal", "params": {"mu": 0.0, "sigma": 0.5}}
#
# - truncnorm: 截断正态分布，参数为min(下限)、max(上限)、mu(均值)和sigma(标准差)
#   例: {"type": "truncnorm", "params": {"min": 0.0, "max": 1.0, "mu": 0.5, "sigma": 0.2}}

# 各模板的先验分布设置，每个模板可以设置不同的先验分布
PRIORS = [
    # 模板的先验分布
    {
        "type": "truncnorm",
        "params": {"min": 0, "max": 1, "mu": 0.0001, "sigma": 0.0002},
    },
    {
        "type": "truncnorm",
        "params": {"min": 0, "max": 10, "mu": 0.2, "sigma": 0.2},
    },
    {
        "type": "truncnorm",
        "params": {"min": 0, "max": 10, "mu": 0.1, "sigma": 0.2},
    },
    {
        "type": "truncnorm",
        "params": {"min": 0, "max": 10, "mu": 0.01, "sigma": 0.02},
    },
    {
        "type": "truncnorm",
        "params": {"min": 0, "max": 10, "mu": 0.0001, "sigma": 0.0005},
    },
    # {"type": "uniform", "params": {"min": 0.0, "max": 1.0}},
    {
        "type": "truncnorm",
        "params": {"min": 0, "max": 10, "mu": 0.02, "sigma": 0.04},
    },
    {
        "type": "truncnorm",
        "params": {"min": 0, "max": 20, "mu": 1, "sigma": 2},
    },
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
