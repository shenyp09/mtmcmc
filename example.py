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

示例脚本
-------
生成合成数据并运行MCMC分析的示例脚本
"""

import os
import numpy as np
from pathlib import Path
import shutil

from data_loader import generate_synthetic_data
from model import log_probability
from mcmc_sampler import run_mcmc
from analyzer import analyze_results, calculate_fit_statistics
from visualizer import create_plots
from html_reporter import generate_html_report

# 导入配置
import config


def main():
    """主函数"""
    # 设置随机种子，确保结果可复现
    np.random.seed(42)

    print("生成合成数据...")
    # 生成合成数据
    n_points = 1000  # 使用较少的点以加快示例运行速度
    n_templates = 3
    true_weights = [1.5, 2.0, 1.0]  # 预设的真实权重

    y, ysigma, templates, templates_sigma, _ = generate_synthetic_data(
        n_points=n_points, n_templates=n_templates, weights=true_weights, seed=42
    )

    print(f"真实权重: {true_weights}")

    # 创建输出目录
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # 保存合成数据到文件
    print("保存合成数据到文件...")

    # 创建data目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # 创建能量轴
    energy = np.linspace(0, 100, n_points)

    # 保存目标能谱
    target_file = "data/synthetic_target.txt"
    np.savetxt(target_file, np.column_stack((energy, y, ysigma)), fmt="%.6f")

    # 临时保存配置中指定的目标文件路径
    original_target = config.TARGET_SPECTRUM

    # 保存模板能谱
    template_files = []
    for i in range(n_templates):
        template_file = f"data/synthetic_template{i+1}.txt"
        np.savetxt(
            template_file,
            np.column_stack((energy, templates[i], templates_sigma[i])),
            fmt="%.6f",
        )
        template_files.append(template_file)

    # 临时保存配置中指定的模板文件路径
    original_templates = config.TEMPLATE_SPECTRA.copy()

    # 临时修改配置，使用生成的合成数据
    config.TARGET_SPECTRUM = target_file
    config.TEMPLATE_SPECTRA = template_files

    # 对于示例，使用较少的MCMC步数以加快速度
    original_nsteps = config.NSTEPS
    original_burnin = config.BURNIN
    config.NSTEPS = 1000
    config.BURNIN = 200

    print("运行MCMC分析...")
    # 运行MCMC
    sampler, pos, prob, state = run_mcmc(
        y,
        ysigma,
        templates,
        templates_sigma,
        ndim=n_templates,
        nwalkers=config.NWALKERS,
        nsteps=config.NSTEPS,
        burnin=config.BURNIN,
        priors=config.PRIORS,
        default_prior=config.DEFAULT_PRIOR,
        progress=config.PROGRESS,
    )

    print("分析结果...")
    # 分析MCMC结果
    results = analyze_results(sampler, burnin=0)

    # 计算拟合统计量
    fit_stats = calculate_fit_statistics(y, ysigma, templates, templates_sigma, results)

    # 输出参数估计值
    print("\n参数估计值:")
    for i in range(n_templates):
        true_val = true_weights[i]
        est_val = results["medians"][i]
        lower = results["lower_bounds"][i]
        upper = results["upper_bounds"][i]
        prior_type = (
            config.PRIORS[i]["type"]
            if i < len(config.PRIORS)
            else config.DEFAULT_PRIOR["type"]
        )
        print(
            f"参数 {i+1} (先验: {prior_type}): {est_val:.4f} [{lower:.4f}, {upper:.4f}] (真实值: {true_val:.4f})"
        )

    print("\n拟合统计量:")
    print(f"卡方值: {fit_stats['chi_square']:.2f}")
    print(f"简化卡方: {fit_stats['reduced_chi_square']:.3f}")
    print(f"自由度: {fit_stats['dof']}")
    print(f"p值: {fit_stats['p_value']:.4f}")

    print("\n使用配置文件运行主脚本的命令:")
    print("python mtmcmc.py")


if __name__ == "__main__":
    main()
