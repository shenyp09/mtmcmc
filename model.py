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

模型模块
-------
定义多模板拟合的模型函数和先验分布

先验分布类型:
- uniform: 均匀分布，参数为min和max
  例: {"type": "uniform", "params": {"min": 0.0, "max": 10.0}}

- normal: 正态分布，参数为mu(均值)和sigma(标准差)
  例: {"type": "normal", "params": {"mu": 1.0, "sigma": 0.5}}

- lognormal: 对数正态分布，参数为mu(对数均值)和sigma(对数标准差)
  例: {"type": "lognormal", "params": {"mu": 0.0, "sigma": 0.5}}

- truncnorm: 截断正态分布，参数为min(下限)、max(上限)、mu(均值)和sigma(标准差)
  例: {"type": "truncnorm", "params": {"min": 0.0, "max": 1.0, "mu": 0.5, "sigma": 0.2}}
"""

import numpy as np
from scipy import stats


def model_predict(params, templates):
    """
    根据给定参数计算模型预测值

    参数:
        params (array): 模型参数数组，表示各模板的权重
        templates (list): 模板能谱列表

    返回:
        array: 模型预测的能谱
    """
    prediction = np.zeros_like(templates[0])

    for i, template in enumerate(templates):
        prediction += params[i] * template

    return prediction


def calculate_total_sigma(params, ysigma, templates_sigma):
    """
    计算考虑所有误差源的总标准差

    参数:
        params (array): 模型参数数组，表示各模板的权重
        ysigma (array): 目标能谱标准差
        templates_sigma (list): 模板能谱标准差列表

    返回:
        array: 总标准差
    """
    # 初始化为目标能谱误差的平方
    total_variance = ysigma**2

    # 加上各模板权重乘以模板误差的平方和
    for i, template_sigma in enumerate(templates_sigma):
        total_variance += (params[i] * template_sigma) ** 2

    return np.sqrt(total_variance)


def log_prior_single(param, prior_type="uniform", prior_params=None):
    """
    计算单个参数的对数先验概率

    参数:
        param (float): 参数值
        prior_type (str): 先验分布类型 {'uniform', 'normal', 'lognormal', 'truncnorm'}
        prior_params (dict): 先验分布参数

    返回:
        float: 对数先验概率
    """
    if prior_params is None:
        prior_params = {}

    # 检查参数是否为正数（权重必须为正）
    if param < 0:
        return -np.inf

    if prior_type == "uniform":
        # 均匀先验，默认为[0, 10]
        min_val = prior_params.get("min", 0.0)
        max_val = prior_params.get("max", 10.0)

        if param < min_val or param > max_val:
            return -np.inf

        # 均匀分布的对数概率是常数，可以省略
        return 0.0

    elif prior_type == "normal":
        # 正态先验，默认为均值1，标准差1
        mu = prior_params.get("mu", 1.0)
        sigma = prior_params.get("sigma", 1.0)

        return stats.norm.logpdf(param, loc=mu, scale=sigma)

    elif prior_type == "lognormal":
        # 对数正态先验，默认为均值0，标准差0.5
        mu = prior_params.get("mu", 0.0)
        sigma = prior_params.get("sigma", 0.5)

        return stats.lognorm.logpdf(param, s=sigma, scale=np.exp(mu))

    elif prior_type == "truncnorm":
        # 截断正态先验，默认为均值0.5，标准差0.5，下限0，上限10
        mu = prior_params.get("mu", 0.5)
        sigma = prior_params.get("sigma", 0.5)
        min_val = prior_params.get("min", 0.0)
        max_val = prior_params.get("max", 10.0)

        # 检查参数是否在范围内
        if param < min_val or param > max_val:
            return -np.inf

        # 计算截断正态分布参数
        a = (min_val - mu) / sigma
        b = (max_val - mu) / sigma

        return stats.truncnorm.logpdf(param, a, b, loc=mu, scale=sigma)

    # 默认情况
    return 0.0


def log_prior(params, priors=None, default_prior=None):
    """
    计算多个参数的对数先验概率，支持为每个参数设置不同的先验分布

    参数:
        params (array): 模型参数数组
        priors (list): 先验分布列表，每个元素是一个包含type和params的字典
        default_prior (dict): 默认先验分布，当priors中没有为特定参数指定先验时使用

    返回:
        float: 对数先验概率之和
    """
    # 如果没有提供先验设置，使用默认均匀先验
    if priors is None:
        priors = []

    if default_prior is None:
        default_prior = {"type": "uniform", "params": {"min": 0.0, "max": 10.0}}

    # 计算总的对数先验
    log_prob = 0.0

    for i, param in enumerate(params):
        # 获取当前参数的先验设置
        if i < len(priors):
            prior = priors[i]
        else:
            prior = default_prior

        # 计算当前参数的对数先验
        param_log_prior = log_prior_single(
            param, prior_type=prior["type"], prior_params=prior["params"]
        )

        # 如果有一个参数的先验为-inf，整个先验为-inf
        if not np.isfinite(param_log_prior):
            return -np.inf

        log_prob += param_log_prior

    return log_prob


def log_likelihood(params, y, ysigma, templates, templates_sigma):
    """
    计算参数的对数似然概率

    参数:
        params (array): 模型参数数组
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表

    返回:
        float: 对数似然概率
    """
    # 计算模型预测值
    prediction = model_predict(params, templates)

    # 计算总标准差
    total_sigma = calculate_total_sigma(params, ysigma, templates_sigma)

    # 计算对数似然（假设正态分布误差）
    log_prob = -0.5 * np.sum(
        ((y - prediction) / total_sigma) ** 2 + np.log(2 * np.pi * total_sigma**2)
    )

    return log_prob


def log_probability(
    params, y, ysigma, templates, templates_sigma, priors=None, default_prior=None
):
    """
    计算参数的对数后验概率 (对数先验 + 对数似然)

    参数:
        params (array): 模型参数数组
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表
        priors (list): 先验分布列表
        default_prior (dict): 默认先验分布

    返回:
        float: 对数后验概率
    """
    # 计算先验
    lp = log_prior(params, priors=priors, default_prior=default_prior)

    # 如果先验为-inf，直接返回
    if not np.isfinite(lp):
        return -np.inf

    # 计算似然并返回后验
    return lp + log_likelihood(params, y, ysigma, templates, templates_sigma)


def parse_prior_params(prior_type, prior_params_str):
    """
    解析先验参数字符串

    参数:
        prior_type (str): 先验分布类型
        prior_params_str (str): 先验参数字符串，格式为'param1,param2,...'

    返回:
        dict: 先验参数字典
    """
    if prior_params_str is None:
        return None

    params = prior_params_str.split(",")

    if prior_type == "uniform":
        if len(params) >= 2:
            return {"min": float(params[0]), "max": float(params[1])}

    elif prior_type == "normal" or prior_type == "lognormal":
        if len(params) >= 2:
            return {"mu": float(params[0]), "sigma": float(params[1])}

    elif prior_type == "truncnorm":
        if len(params) >= 4:
            return {
                "min": float(params[0]),
                "max": float(params[1]),
                "mu": float(params[2]),
                "sigma": float(params[3]),
            }
        elif len(params) >= 2:
            # 如果只提供了min和max，使用范围中点作为均值
            min_val = float(params[0])
            max_val = float(params[1])
            mu = (min_val + max_val) / 2
            sigma = (max_val - min_val) / 4  # 默认使区间覆盖约95%置信区间
            return {"min": min_val, "max": max_val, "mu": mu, "sigma": sigma}

    # 默认返回None，使用默认参数
    return None
