#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块
--------
用于生成各种可视化图表，包括后验分布、拟合结果等
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
from pathlib import Path
import os

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from model import model_predict, calculate_total_sigma
from analyzer import error_contribution_analysis


def set_matplotlib_style():
    """设置Matplotlib样式"""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_context("talk")
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def create_plots(
    y,
    ysigma,
    templates,
    templates_sigma,
    results,
    fit_stats,
    output_dir="results",
    interactive=False,
):
    """
    创建所有图表

    参数:
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表
        results (dict): MCMC分析结果
        fit_stats (dict): 拟合统计量
        output_dir (str): 输出目录
        interactive (bool): 是否创建交互式图表

    返回:
        dict: 图表文件路径字典
    """
    # 设置Matplotlib样式
    set_matplotlib_style()

    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_files = {}

    # 创建后验分布图
    corner_fig_path = os.path.join(output_dir, "posterior_corner.png")
    create_corner_plot(results, corner_fig_path)
    figure_files["corner"] = corner_fig_path

    # 创建链迹图
    trace_fig_path = os.path.join(output_dir, "chain_traces.png")
    create_trace_plot(results, trace_fig_path)
    figure_files["trace"] = trace_fig_path

    # 创建拟合结果图
    fit_fig_path = os.path.join(output_dir, "fit_results.png")
    create_fit_plot(
        y, ysigma, templates, templates_sigma, results, fit_stats, fit_fig_path
    )
    figure_files["fit"] = fit_fig_path

    # 创建残差图
    residual_fig_path = os.path.join(output_dir, "residuals.png")
    create_residual_plot(fit_stats, residual_fig_path)
    figure_files["residual"] = residual_fig_path

    # 创建模板贡献图
    contribution_fig_path = os.path.join(output_dir, "template_contributions.png")
    create_contribution_plot(templates, fit_stats, contribution_fig_path)
    figure_files["contribution"] = contribution_fig_path

    # 创建误差贡献分析图
    error_analysis = error_contribution_analysis(
        results["map_estimate"], ysigma, templates_sigma
    )
    error_fig_path = os.path.join(output_dir, "error_contributions.png")
    create_error_contribution_plot(error_analysis, error_fig_path)
    figure_files["error"] = error_fig_path

    # 如果启用交互式图表且Plotly可用，创建交互式图表
    if interactive and PLOTLY_AVAILABLE:
        # 交互式后验分布图
        interactive_corner_path = os.path.join(output_dir, "interactive_posterior.html")
        create_interactive_corner(results, interactive_corner_path)
        figure_files["interactive_corner"] = interactive_corner_path

        # 交互式拟合结果图
        interactive_fit_path = os.path.join(output_dir, "interactive_fit.html")
        create_interactive_fit(
            y,
            ysigma,
            templates,
            templates_sigma,
            results,
            fit_stats,
            interactive_fit_path,
        )
        figure_files["interactive_fit"] = interactive_fit_path

        # 交互式链迹图
        interactive_trace_path = os.path.join(output_dir, "interactive_traces.html")
        create_interactive_trace(results, interactive_trace_path)
        figure_files["interactive_trace"] = interactive_trace_path

    return figure_files


def create_corner_plot(results, output_path):
    """创建参数后验分布角图"""
    samples = results["samples"]
    ndim = samples.shape[1]

    # 创建标签
    labels = [f"参数 {i+1}" for i in range(ndim)]

    # 设置图表大小
    plt.figure(figsize=(12, 10))

    # 创建角图
    corner_fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt=".3f",
    )

    # 添加标题
    corner_fig.suptitle("参数后验分布", fontsize=16, y=1.02)

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_trace_plot(results, output_path):
    """创建MCMC链迹图"""
    samples = results["samples"]
    ndim = samples.shape[1]

    # 创建图表网格
    fig, axes = plt.subplots(ndim, figsize=(12, 2 * ndim))

    # 对每个参数绘制链迹
    for i in range(ndim):
        ax = axes[i] if ndim > 1 else axes
        ax.plot(samples[:, i], "k-", alpha=0.3)
        ax.set_ylabel(f"参数 {i+1}")

        # 添加68%置信区间
        ax.axhline(results["medians"][i], color="r")
        ax.axhline(results["lower_bounds"][i], color="r", linestyle="--")
        ax.axhline(results["upper_bounds"][i], color="r", linestyle="--")

    # 添加标题
    fig.suptitle("MCMC链迹图", fontsize=16)
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_fit_plot(
    y, ysigma, templates, templates_sigma, results, fit_stats, output_path
):
    """创建拟合结果图"""
    # 获取参数最佳估计值
    map_params = results["map_estimate"]

    # 获取预测值和残差
    prediction = fit_stats["prediction"]
    residuals = fit_stats["residuals"]

    # 计算总标准差
    total_sigma = calculate_total_sigma(map_params, ysigma, templates_sigma)

    # 创建x轴 (通道/能量)
    x = np.arange(len(y))

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制数据点及误差棒
    ax.errorbar(x, y, yerr=ysigma, fmt="o", color="blue", alpha=0.5, label="数据")

    # 绘制拟合曲线
    ax.plot(x, prediction, "r-", label="拟合")

    # 绘制总误差带
    ax.fill_between(
        x,
        prediction - total_sigma,
        prediction + total_sigma,
        color="red",
        alpha=0.2,
        label="$1\\sigma$ 误差带",
    )

    # 绘制各模板贡献
    for i, template in enumerate(templates):
        template_contribution = map_params[i] * template
        ax.plot(x, template_contribution, "--", alpha=0.7, label=f"模板 {i+1}")

    # 设置图表属性
    ax.set_xlabel("通道")
    ax.set_ylabel("计数")
    ax.set_title("能谱拟合结果")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 添加拟合统计信息文本框
    stats_text = (
        f"$\\chi^2$ = {fit_stats['chi_square']:.2f}\n"
        f"简化 $\\chi^2$ = {fit_stats['reduced_chi_square']:.2f}\n"
        f"自由度 = {fit_stats['dof']}\n"
        f"p值 = {fit_stats['p_value']:.4f}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_residual_plot(fit_stats, output_path):
    """创建残差图"""
    # 获取残差和归一化残差
    residuals = fit_stats["residuals"]
    normalized_residuals = fit_stats["normalized_residuals"]

    # 创建x轴 (通道/能量)
    x = np.arange(len(residuals))

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制残差
    axes[0].plot(x, residuals, "o", color="blue", alpha=0.5)
    axes[0].axhline(y=0, color="r", linestyle="-")
    axes[0].set_ylabel("残差")
    axes[0].set_title("拟合残差")
    axes[0].grid(True, alpha=0.3)

    # 绘制归一化残差
    axes[1].plot(x, normalized_residuals, "o", color="green", alpha=0.5)
    axes[1].axhline(y=0, color="r", linestyle="-")
    axes[1].axhline(y=1, color="r", linestyle="--")
    axes[1].axhline(y=-1, color="r", linestyle="--")
    axes[1].set_xlabel("通道")
    axes[1].set_ylabel("归一化残差")
    axes[1].set_title("归一化残差 (残差/σ)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_contribution_plot(templates, fit_stats, output_path):
    """创建模板贡献饼图"""
    # 获取模板贡献
    template_contributions = fit_stats["template_contributions"]

    # 提取贡献百分比和标签
    percentages = [tc["percentage"] for tc in template_contributions]
    labels = [
        f'模板 {tc["index"]+1} ({tc["percentage"]:.1f}%)'
        for tc in template_contributions
    ]

    # 创建饼图
    plt.figure(figsize=(10, 8))
    plt.pie(percentages, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
    plt.axis("equal")  # 确保饼图是圆的
    plt.title("各模板对拟合结果的贡献比例")

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_error_contribution_plot(error_analysis, output_path):
    """创建误差贡献分析图"""
    # 提取误差贡献信息
    target_contribution = error_analysis["target_contribution"]["mean"]
    template_contributions = [
        tc["mean_contribution"] for tc in error_analysis["template_contributions"]
    ]

    # 创建标签
    labels = ["目标谱"] + [
        f'模板 {tc["index"]+1}' for tc in error_analysis["template_contributions"]
    ]

    # 创建值
    values = [target_contribution] + template_contributions

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        labels, values, color=["blue"] + ["orange"] * len(template_contributions)
    )

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.xlabel("误差源")
    plt.ylabel("平均贡献百分比 (%)")
    plt.title("各误差源对总误差的平均贡献")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_interactive_corner(results, output_path):
    """创建交互式参数后验分布图"""
    if not PLOTLY_AVAILABLE:
        return

    samples = results["samples"]
    ndim = samples.shape[1]

    # 创建标签
    labels = [f"参数 {i+1}" for i in range(ndim)]

    # 创建子图网格
    fig = make_subplots(
        rows=ndim,
        cols=ndim,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            f"{labels[i]} vs {labels[j]}" if i != j else f"{labels[i]} 分布"
            for i in range(ndim)
            for j in range(ndim)
        ],
    )

    # 填充子图
    for i in range(ndim):
        for j in range(ndim):
            if i == j:  # 对角线上绘制直方图
                fig.add_trace(
                    go.Histogram(x=samples[:, i], nbinsx=30, name=labels[i]),
                    row=i + 1,
                    col=j + 1,
                )
            elif i > j:  # 对角线下方绘制散点图
                fig.add_trace(
                    go.Scatter(
                        x=samples[:, j],
                        y=samples[:, i],
                        mode="markers",
                        marker=dict(size=2, opacity=0.5),
                        name=f"{labels[j]} vs {labels[i]}",
                    ),
                    row=i + 1,
                    col=j + 1,
                )

    # 更新布局
    fig.update_layout(
        title="交互式参数后验分布",
        height=250 * ndim,
        width=250 * ndim,
        showlegend=False,
    )

    # 保存为HTML文件
    fig.write_html(output_path)


def create_interactive_fit(
    y, ysigma, templates, templates_sigma, results, fit_stats, output_path
):
    """创建交互式拟合结果图"""
    if not PLOTLY_AVAILABLE:
        return

    # 获取参数最佳估计值
    map_params = results["map_estimate"]

    # 获取预测值和残差
    prediction = fit_stats["prediction"]
    residuals = fit_stats["residuals"]
    normalized_residuals = fit_stats["normalized_residuals"]

    # 计算总标准差
    total_sigma = calculate_total_sigma(map_params, ysigma, templates_sigma)

    # 创建x轴 (通道/能量)
    x = np.arange(len(y))

    # 创建子图
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=("能谱拟合", "归一化残差")
    )

    # 添加数据点
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="数据",
            error_y=dict(type="data", array=ysigma, visible=True),
        ),
        row=1,
        col=1,
    )

    # 添加拟合曲线
    fig.add_trace(
        go.Scatter(
            x=x, y=prediction, mode="lines", name="拟合", line=dict(color="red")
        ),
        row=1,
        col=1,
    )

    # 添加误差带
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate(
                [prediction + total_sigma, (prediction - total_sigma)[::-1]]
            ),
            fill="toself",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="rgba(255,0,0,0)"),
            name="$1\\sigma$ 误差带",
        ),
        row=1,
        col=1,
    )

    # 添加各模板贡献
    for i, template in enumerate(templates):
        template_contribution = map_params[i] * template
        fig.add_trace(
            go.Scatter(
                x=x,
                y=template_contribution,
                mode="lines",
                name=f"模板 {i+1}",
                line=dict(dash="dash"),
            ),
            row=1,
            col=1,
        )

    # 添加归一化残差
    fig.add_trace(
        go.Scatter(
            x=x,
            y=normalized_residuals,
            mode="markers",
            name="归一化残差",
            marker=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    # 添加零线和±1σ线
    fig.add_trace(
        go.Scatter(
            x=x, y=np.zeros_like(x), mode="lines", name="零线", line=dict(color="red")
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.ones_like(x),
            mode="lines",
            name="+1σ",
            line=dict(color="red", dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=-np.ones_like(x),
            mode="lines",
            name="-1σ",
            line=dict(color="red", dash="dash"),
        ),
        row=2,
        col=1,
    )

    # 更新布局
    fig.update_layout(
        title="交互式能谱拟合结果",
        height=800,
        width=1000,
        xaxis2_title="通道",
        yaxis_title="计数",
        yaxis2_title="残差/σ",
    )

    # 添加拟合统计信息
    stats_text = (
        f"$\\chi^2$ = {fit_stats['chi_square']:.2f}<br>"
        f"简化 $\\chi^2$ = {fit_stats['reduced_chi_square']:.2f}<br>"
        f"自由度 = {fit_stats['dof']}<br>"
        f"p值 = {fit_stats['p_value']:.4f}"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=stats_text,
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
    )

    # 保存为HTML文件
    fig.write_html(output_path)


def create_interactive_trace(results, output_path):
    """创建交互式MCMC链迹图"""
    if not PLOTLY_AVAILABLE:
        return

    samples = results["samples"]
    ndim = samples.shape[1]

    # 创建子图
    fig = make_subplots(
        rows=ndim,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"参数 {i+1}" for i in range(ndim)],
    )

    # 为每个参数添加链迹
    for i in range(ndim):
        fig.add_trace(
            go.Scatter(
                y=samples[:, i],
                mode="lines",
                name=f"参数 {i+1}",
                line=dict(color="black", width=1),
            ),
            row=i + 1,
            col=1,
        )

        # 添加中位数线
        fig.add_trace(
            go.Scatter(
                y=[results["medians"][i]] * len(samples),
                mode="lines",
                name=f"中位数",
                line=dict(color="red"),
            ),
            row=i + 1,
            col=1,
        )

        # 添加置信区间线
        fig.add_trace(
            go.Scatter(
                y=[results["lower_bounds"][i]] * len(samples),
                mode="lines",
                name=f"16%分位数",
                line=dict(color="red", dash="dash"),
            ),
            row=i + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                y=[results["upper_bounds"][i]] * len(samples),
                mode="lines",
                name=f"84%分位数",
                line=dict(color="red", dash="dash"),
            ),
            row=i + 1,
            col=1,
        )

    # 更新布局
    fig.update_layout(
        title="交互式MCMC链迹图", height=250 * ndim, width=1000, showlegend=False
    )

    # 保存为HTML文件
    fig.write_html(output_path)
