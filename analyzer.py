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

分析模块
-------
用于分析MCMC采样结果，计算各种统计量
"""

import numpy as np
from scipy import stats

from mcmc_sampler import get_flat_samples, check_convergence
from model import model_predict, calculate_total_sigma


def analyze_results(sampler, burnin=0, thin=1):
    """
    分析MCMC采样结果

    参数:
        sampler (emcee.EnsembleSampler): MCMC采样器
        burnin (int): 预热步数
        thin (int): 抽稀间隔

    返回:
        dict: 结果分析字典，包含
            - samples: 展平的样本
            - medians: 参数中位数
            - lower_bounds: 参数下界（16%分位数）
            - upper_bounds: 参数上界（84%分位数）
            - means: 参数均值
            - stds: 参数标准差
            - map_estimate: 最大后验估计
            - log_probability: 最大后验对应的对数概率
            - autocorr_time: 自相关时间
            - effective_samples: 有效样本数
            - convergence: 收敛信息
    """
    # 获取展平的样本
    samples = get_flat_samples(sampler, burnin=burnin, thin=thin)

    # 参数数量
    ndim = samples.shape[1]

    # 计算统计量
    medians = np.median(samples, axis=0)
    lower_bounds = np.percentile(samples, 16, axis=0)
    upper_bounds = np.percentile(samples, 84, axis=0)
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)

    # 找到最大后验估计
    log_probs = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    max_prob_idx = np.argmax(log_probs)
    map_estimate = samples[max_prob_idx]
    max_log_prob = log_probs[max_prob_idx]

    # 检查收敛性
    converged, tau = check_convergence(sampler, burnin=burnin)

    # 计算有效样本数
    if tau is not None:
        effective_samples = sampler.iteration // tau
    else:
        effective_samples = None

    # 收敛信息
    convergence_info = {
        "converged": converged,
        "autocorr_time": tau,
        "effective_samples": effective_samples,
        "threshold_ratio": (
            np.nan if tau is None else sampler.iteration / (50 * np.max(tau))
        ),
    }

    # 构建结果字典
    results = {
        "samples": samples,
        "medians": medians,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "means": means,
        "stds": stds,
        "map_estimate": map_estimate,
        "log_probability": max_log_prob,
        "autocorr_time": tau,
        "effective_samples": effective_samples,
        "convergence": convergence_info,
    }

    return results


def calculate_fit_statistics(y, ysigma, templates, templates_sigma, results):
    """
    计算拟合统计量

    参数:
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表
        results (dict): MCMC分析结果

    返回:
        dict: 拟合统计量字典
    """
    # 获取最大后验参数估计值
    map_params = results["map_estimate"]

    # 计算预测值
    prediction = model_predict(map_params, templates)

    # 计算总标准差
    total_sigma = calculate_total_sigma(map_params, ysigma, templates_sigma)

    # 计算残差
    residuals = y - prediction

    # 计算归一化残差
    normalized_residuals = residuals / total_sigma

    # 计算卡方值
    chi_square = np.sum(normalized_residuals**2)

    # 自由度 (数据点数 - 参数数)
    dof = len(y) - len(map_params)

    # 计算简化卡方
    reduced_chi_square = chi_square / dof

    # 计算卡方p值
    p_value = 1.0 - stats.chi2.cdf(chi_square, dof)

    # 计算BIC (贝叶斯信息准则)
    # BIC = -2 * log(L) + k * log(n)
    # 其中L是似然函数，k是参数数量，n是数据点数量
    log_likelihood = -0.5 * chi_square  # 简化计算
    bic = -2 * log_likelihood + len(map_params) * np.log(len(y))

    # 计算AIC (赤池信息准则)
    # AIC = 2k - 2log(L)
    aic = 2 * len(map_params) - 2 * log_likelihood

    # 计算各模板的贡献比例
    total_counts = np.sum(prediction)
    template_contributions = []
    for i, template in enumerate(templates):
        template_contribution = map_params[i] * template
        contribution_percentage = 100 * np.sum(template_contribution) / total_counts
        template_contributions.append(
            {
                "index": i,
                "weight": map_params[i],
                "counts": np.sum(template_contribution),
                "percentage": contribution_percentage,
            }
        )

    # 构建拟合统计量字典
    stats_dict = {
        "prediction": prediction,
        "residuals": residuals,
        "normalized_residuals": normalized_residuals,
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "dof": dof,
        "p_value": p_value,
        "bic": bic,
        "aic": aic,
        "template_contributions": template_contributions,
    }

    return stats_dict


def error_contribution_analysis(map_params, ysigma, templates_sigma):
    """
    分析不同误差源对总误差的贡献

    参数:
        map_params (array): 模型参数最佳估计值
        ysigma (array): 目标能谱标准差
        templates_sigma (list): 模板能谱标准差列表

    返回:
        dict: 误差贡献分析结果
    """
    # 计算目标谱误差贡献
    target_variance = ysigma**2

    # 计算各模板误差贡献
    template_variances = []
    for i, template_sigma in enumerate(templates_sigma):
        template_var = (map_params[i] * template_sigma) ** 2
        template_variances.append(template_var)

    # 计算总方差
    total_variance = target_variance.copy()
    for var in template_variances:
        total_variance += var

    # 计算各误差源的相对贡献
    target_contribution = 100 * target_variance / total_variance

    template_contributions = []
    for i, var in enumerate(template_variances):
        contribution = 100 * var / total_variance
        template_contributions.append(
            {
                "index": i,
                "weight": map_params[i],
                "mean_contribution": np.mean(contribution),
                "max_contribution": np.max(contribution),
            }
        )

    # 构建误差分析结果
    error_analysis = {
        "target_contribution": {
            "mean": np.mean(target_contribution),
            "max": np.max(target_contribution),
        },
        "template_contributions": template_contributions,
        "target_variance": target_variance,
        "template_variances": template_variances,
        "total_variance": total_variance,
    }

    return error_analysis
