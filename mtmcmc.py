#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi Template MCMC Bayesian Analysis
-------------------------------------
用于能谱拟合的多模板MCMC贝叶斯分析算法

用法: python mtmcmc.py

该程序使用config.py中的配置参数。
"""

import os
import sys
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from data_loader import load_data, preprocess_data
from model import log_probability
from mcmc_sampler import run_mcmc
from analyzer import analyze_results, calculate_fit_statistics
from visualizer import create_plots
from html_reporter import generate_html_report

# 导入配置
import config


def main():
    """主函数"""
    # 创建输出目录（添加时间戳）
    output_dir = Path(config.OUTPUT_DIR)
    if hasattr(config, "ADD_TIMESTAMP") and config.ADD_TIMESTAMP:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在加载数据...")
    # 加载目标谱和模板谱数据
    target_data = load_data(config.TARGET_SPECTRUM)
    template_data_list = [load_data(template) for template in config.TEMPLATE_SPECTRA]

    # 预处理数据
    y, ysigma, templates, templates_sigma = preprocess_data(
        target_data, template_data_list, error_handling=config.ERROR_HANDLING
    )

    # 设置模型参数
    ndim = len(template_data_list)  # 参数维度 = 模板数量
    nwalkers = config.NWALKERS
    nsteps = config.NSTEPS
    burnin = config.BURNIN

    # 确保PRIORS列表长度与模板数量一致
    priors = config.PRIORS
    if len(priors) < ndim:
        print(
            f"警告: 配置中的先验分布数量({len(priors)})少于模板数量({ndim})，将对剩余参数使用默认先验"
        )

    print(f"正在运行MCMC采样 ({ndim}个参数, {nwalkers}个walkers, {nsteps}步)...")
    start_time = time.time()

    # 运行MCMC采样
    sampler, pos, prob, state = run_mcmc(
        y,
        ysigma,
        templates,
        templates_sigma,
        ndim=ndim,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        priors=priors,
        default_prior=config.DEFAULT_PRIOR,
        progress=config.PROGRESS,
        ncores=config.NCORES,
    )

    end_time = time.time()
    print(f"MCMC采样完成，耗时 {end_time - start_time:.2f} 秒")

    print("正在分析结果...")
    # 分析MCMC结果
    results = analyze_results(sampler, burnin)

    # 计算拟合统计量
    fit_stats = calculate_fit_statistics(y, ysigma, templates, templates_sigma, results)

    print("正在生成图表...")
    # 创建可视化图表
    figure_files = create_plots(
        y,
        ysigma,
        templates,
        templates_sigma,
        results,
        fit_stats,
        output_dir=output_dir,
        interactive=config.INTERACTIVE_PLOTS,
    )

    # 生成HTML报告
    if config.HTML_REPORT:
        print("正在生成HTML报告...")
        html_files = generate_html_report(
            y,
            ysigma,
            templates,
            templates_sigma,
            results,
            fit_stats,
            figure_files,
            template_dir=config.TEMPLATE_DIR,
            output_dir=output_dir,
            interactive=config.INTERACTIVE_PLOTS,
            languages=config.HTML_LANGUAGES,
        )

        # 打印各语言版本HTML报告的路径
        for lang, html_file in html_files.items():
            lang_name = "中文" if lang == "zh" else "英文"
            print(f"{lang_name}HTML报告已生成: {html_file}")

    print(f"所有结果已保存至 {output_dir}")

    # 打印参数估计结果
    print("\n模板权重估计结果:")
    for i, template in enumerate(config.TEMPLATE_SPECTRA):
        template_name = os.path.basename(template)
        median = results["medians"][i]
        lower = results["lower_bounds"][i]
        upper = results["upper_bounds"][i]
        print(f"  {template_name}: {median:.4f} [{lower:.4f}, {upper:.4f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
