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


def set_matplotlib_style(lang="zh"):
    """
    设置Matplotlib样式

    参数:
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_context("talk")

    if lang == "zh":
        # 中文设置
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
    else:
        # 英文设置
        plt.rcParams["font.sans-serif"] = ["Arial"]
        plt.rcParams["axes.unicode_minus"] = True


def create_plots(
    y,
    ysigma,
    templates,
    templates_sigma,
    results,
    fit_stats,
    output_dir="results",
    interactive=False,
    lang="zh",
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
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)

    返回:
        dict: 图表文件路径字典
    """
    # 设置Matplotlib样式
    set_matplotlib_style(lang)

    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_files = {}

    # 为文件名添加语言标识
    lang_suffix = f"_{lang}" if lang != "zh" else ""

    # 创建后验分布图
    corner_fig_path = os.path.join(output_dir, f"posterior_corner{lang_suffix}.png")
    create_corner_plot(results, corner_fig_path, lang)
    figure_files["corner"] = corner_fig_path

    # 创建链迹图
    trace_fig_path = os.path.join(output_dir, f"chain_traces{lang_suffix}.png")
    create_trace_plot(results, trace_fig_path, lang)
    figure_files["trace"] = trace_fig_path

    # 创建拟合结果图
    fit_fig_path = os.path.join(output_dir, f"fit_results{lang_suffix}.png")
    create_fit_plot(
        y, ysigma, templates, templates_sigma, results, fit_stats, fit_fig_path, lang
    )
    figure_files["fit"] = fit_fig_path

    # 创建残差图
    residual_fig_path = os.path.join(output_dir, f"residuals{lang_suffix}.png")
    create_residual_plot(fit_stats, residual_fig_path, lang)
    figure_files["residual"] = residual_fig_path

    # 创建模板贡献图
    contribution_fig_path = os.path.join(
        output_dir, f"template_contributions{lang_suffix}.png"
    )
    create_contribution_plot(templates, fit_stats, contribution_fig_path, lang)
    figure_files["contribution"] = contribution_fig_path

    # 创建误差贡献分析图
    error_analysis = error_contribution_analysis(
        results["map_estimate"], ysigma, templates_sigma
    )
    error_fig_path = os.path.join(output_dir, f"error_contributions{lang_suffix}.png")
    create_error_contribution_plot(error_analysis, error_fig_path, lang)
    figure_files["error"] = error_fig_path

    # 如果启用交互式图表且Plotly可用，创建交互式图表
    if interactive and PLOTLY_AVAILABLE:
        # 交互式后验分布图
        interactive_corner_path = os.path.join(
            output_dir, f"interactive_posterior{lang_suffix}.html"
        )
        create_interactive_corner(results, interactive_corner_path, lang)
        figure_files["interactive_corner"] = interactive_corner_path

        # 交互式拟合结果图
        interactive_fit_path = os.path.join(
            output_dir, f"interactive_fit{lang_suffix}.html"
        )
        create_interactive_fit(
            y,
            ysigma,
            templates,
            templates_sigma,
            results,
            fit_stats,
            interactive_fit_path,
            lang,
        )
        figure_files["interactive_fit"] = interactive_fit_path

        # 交互式链迹图
        interactive_trace_path = os.path.join(
            output_dir, f"interactive_traces{lang_suffix}.html"
        )
        create_interactive_trace(results, interactive_trace_path, lang)
        figure_files["interactive_trace"] = interactive_trace_path

    return figure_files


def create_corner_plot(results, output_path, lang="zh"):
    """
    创建参数后验分布角图

    参数:
        results (dict): MCMC分析结果
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    samples = results["samples"]
    ndim = samples.shape[1]

    # 创建标签
    if lang == "zh":
        labels = [f"参数 {i+1}" for i in range(ndim)]
        title = "参数后验分布"
    else:
        labels = [f"Parameter {i+1}" for i in range(ndim)]
        title = "Parameter Posterior Distribution"

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
    corner_fig.suptitle(title, fontsize=16, y=1.02)

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_trace_plot(results, output_path, lang="zh"):
    """
    创建MCMC链迹图

    参数:
        results (dict): MCMC分析结果
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    samples = results["samples"]
    ndim = samples.shape[1]

    # 创建图表网格
    fig, axes = plt.subplots(ndim, figsize=(12, 2 * ndim))

    # 对每个参数绘制链迹
    for i in range(ndim):
        ax = axes[i] if ndim > 1 else axes
        ax.plot(samples[:, i], "k-", alpha=0.3)

        if lang == "zh":
            ax.set_ylabel(f"参数 {i+1}")
            title = "MCMC链迹图"
        else:
            ax.set_ylabel(f"Parameter {i+1}")
            title = "MCMC Chain Traces"

        # 添加68%置信区间
        ax.axhline(results["medians"][i], color="r")
        ax.axhline(results["lower_bounds"][i], color="r", linestyle="--")
        ax.axhline(results["upper_bounds"][i], color="r", linestyle="--")

    # 添加标题
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_fit_plot(
    y, ysigma, templates, templates_sigma, results, fit_stats, output_path, lang="zh"
):
    """
    创建拟合结果图

    参数:
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表
        results (dict): MCMC分析结果
        fit_stats (dict): 拟合统计量
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    # 获取参数最佳估计值
    map_params = results["map_estimate"]

    # 获取预测值和残差
    prediction = fit_stats["prediction"]
    residuals = fit_stats["residuals"]
    x = np.arange(len(y))

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制数据点及误差棒
    ax.errorbar(
        x,
        y,
        yerr=ysigma,
        fmt="o",
        color="blue",
        alpha=0.5,
        label="Data" if lang == "en" else "数据",
    )

    # 绘制拟合曲线
    ax.plot(
        x,
        prediction,
        "-",
        color="red",
        linewidth=2,
        label="Fit" if lang == "en" else "拟合",
    )

    # 绘制各模板贡献
    for i, template in enumerate(templates):
        ax.plot(
            x,
            map_params[i] * template,
            "--",
            linewidth=1,
            alpha=0.7,
            label=f"Template {i+1}" if lang == "en" else f"模板 {i+1}",
        )

    # 设置图表属性
    if lang == "zh":
        ax.set_title("拟合结果")
        ax.set_xlabel("通道")
        ax.set_ylabel("计数")
    else:
        ax.set_title("Fit Results")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")

    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加拟合统计信息
    chi2 = fit_stats["chi_square"]
    reduced_chi2 = fit_stats["reduced_chi_square"]
    dof = fit_stats["dof"]

    if lang == "zh":
        stats_text = (
            r"$\chi^2 = {0:.2f}$".format(chi2)
            + r"\n$\chi^2_r = {0:.3f}$".format(reduced_chi2)
            + r"\n自由度 = {0}".format(dof)
        )
    else:
        stats_text = (
            r"$\chi^2 = {0:.2f}$".format(chi2)
            + r"\n$\chi^2_r = {0:.3f}$".format(reduced_chi2)
            + r"\nDOF = {0}".format(dof)
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


def create_residual_plot(fit_stats, output_path, lang="zh"):
    """
    创建残差图

    参数:
        fit_stats (dict): 拟合统计量
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    # 获取残差和归一化残差
    residuals = fit_stats["residuals"]
    normalized_residuals = fit_stats["normalized_residuals"]
    x = np.arange(len(residuals))

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制残差
    axes[0].plot(x, residuals, "o", color="blue", alpha=0.5)
    axes[0].axhline(y=0, color="r", linestyle="-")

    if lang == "zh":
        axes[0].set_ylabel("残差")
        title0 = "拟合残差"
    else:
        axes[0].set_ylabel("Residuals")
        title0 = "Fit Residuals"
    axes[0].set_title(title0)
    axes[0].grid(True, alpha=0.3)

    # 绘制归一化残差
    axes[1].plot(x, normalized_residuals, "o", color="blue", alpha=0.5)
    axes[1].axhline(y=0, color="r", linestyle="-")
    axes[1].axhline(y=1, color="r", linestyle="--")
    axes[1].axhline(y=-1, color="r", linestyle="--")

    if lang == "zh":
        axes[1].set_xlabel("通道")
        axes[1].set_ylabel("归一化残差")
        title1 = "归一化残差 (残差/σ)"
    else:
        axes[1].set_xlabel("Channel")
        axes[1].set_ylabel("Normalized Residuals")
        title1 = "Normalized Residuals (Residuals/σ)"
    axes[1].set_title(title1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_contribution_plot(templates, fit_stats, output_path, lang="zh"):
    """
    创建模板贡献饼图

    参数:
        templates (list): 模板能谱列表
        fit_stats (dict): 拟合统计量
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    # 获取模板贡献
    template_contributions = fit_stats["template_contributions"]

    # 提取贡献百分比和标签
    percentages = [tc["percentage"] for tc in template_contributions]

    if lang == "zh":
        labels = [
            f'模板 {tc["index"]+1} ({tc["percentage"]:.1f}%)'
            for tc in template_contributions
        ]
        title = "模板贡献比例"
    else:
        labels = [
            f'Template {tc["index"]+1} ({tc["percentage"]:.1f}%)'
            for tc in template_contributions
        ]
        title = "Template Contribution Ratio"

    # 创建饼图
    plt.figure(figsize=(10, 8))
    plt.pie(
        percentages,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        shadow=True,
        explode=[0.05] * len(percentages),
    )
    plt.axis("equal")
    plt.title(title)

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_error_contribution_plot(error_analysis, output_path, lang="zh"):
    """
    创建误差贡献分析图

    参数:
        error_analysis (dict): 误差贡献分析结果
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    # 提取误差贡献信息
    target_contribution = error_analysis["target_contribution"]["mean"]
    template_contributions = [
        tc["mean_contribution"] for tc in error_analysis["template_contributions"]
    ]
    total_contributions = [target_contribution] + template_contributions

    # 创建标签
    if lang == "zh":
        labels = ["目标谱"] + [
            f'模板 {tc["index"]+1}' for tc in error_analysis["template_contributions"]
        ]
        title = "误差贡献分析"
        ylabel = "平均误差贡献 (%)"
    else:
        labels = ["Target Spectrum"] + [
            f'Template {tc["index"]+1}'
            for tc in error_analysis["template_contributions"]
        ]
        title = "Error Contribution Analysis"
        ylabel = "Average Error Contribution (%)"

    # 创建值
    y_pos = np.arange(len(labels))

    # 创建图表
    plt.figure(figsize=(10, 6))
    bars = plt.bar(y_pos, total_contributions, align="center", alpha=0.7)
    plt.xticks(y_pos, labels)
    plt.ylabel(ylabel)
    plt.title(title)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_interactive_corner(results, output_path, lang="zh"):
    """
    创建交互式参数后验分布图

    参数:
        results (dict): MCMC分析结果
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    if not PLOTLY_AVAILABLE:
        print("警告: Plotly不可用，跳过交互式图表创建")
        return

    samples = results["samples"]
    ndim = samples.shape[1]

    # 创建标签
    if lang == "zh":
        labels = [f"参数 {i+1}" for i in range(ndim)]
        title = "参数后验分布"
    else:
        labels = [f"Parameter {i+1}" for i in range(ndim)]
        title = "Parameter Posterior Distribution"

    # 创建子图网格
    fig = make_subplots(
        rows=ndim,
        cols=ndim,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
    )

    # 添加一维分布和二维分布
    for i in range(ndim):
        for j in range(ndim):
            if i == j:  # 对角线上：添加一维直方图
                fig.add_trace(
                    go.Histogram(
                        x=samples[:, i],
                        name=labels[i],
                        marker=dict(color="royalblue"),
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=j + 1,
                )

                # 添加中位数和68%置信区间
                median = results["medians"][i]
                lower = results["lower_bounds"][i]
                upper = results["upper_bounds"][i]

                fig.add_vline(
                    x=median,
                    line=dict(color="red", width=2),
                    row=i + 1,
                    col=j + 1,
                )
                fig.add_vline(
                    x=lower,
                    line=dict(color="red", width=1.5, dash="dash"),
                    row=i + 1,
                    col=j + 1,
                )
                fig.add_vline(
                    x=upper,
                    line=dict(color="red", width=1.5, dash="dash"),
                    row=i + 1,
                    col=j + 1,
                )
            elif j < i:  # 对角线下方：添加二维散点图
                fig.add_trace(
                    go.Scatter(
                        x=samples[:, j],
                        y=samples[:, i],
                        mode="markers",
                        marker=dict(
                            color="royalblue",
                            size=3,
                            opacity=0.5,
                        ),
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=j + 1,
                )

    # 更新布局
    fig.update_layout(
        title=title,
        width=900,
        height=900,
        showlegend=False,
        plot_bgcolor="white",
    )

    # 更新轴标签和网格
    for i in range(ndim):
        for j in range(ndim):
            if i == ndim - 1:  # 最底行，添加x轴标签
                fig.update_xaxes(
                    title_text=labels[j], row=i + 1, col=j + 1, showgrid=True
                )
            if j == 0:  # 最左列，添加y轴标签
                fig.update_yaxes(
                    title_text=labels[i], row=i + 1, col=j + 1, showgrid=True
                )

    # 保存为HTML文件
    fig.write_html(output_path)


def create_interactive_fit(
    y, ysigma, templates, templates_sigma, results, fit_stats, output_path, lang="zh"
):
    """
    创建交互式拟合结果图

    参数:
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表
        results (dict): MCMC分析结果
        fit_stats (dict): 拟合统计量
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    if not PLOTLY_AVAILABLE:
        print("警告: Plotly不可用，跳过交互式图表创建")
        return

    # 获取参数最佳估计值
    map_params = results["map_estimate"]

    # 获取预测值和残差
    prediction = fit_stats["prediction"]
    residuals = fit_stats["residuals"]
    normalized_residuals = fit_stats["normalized_residuals"]
    x = np.arange(len(y))

    # 创建子图
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "拟合结果" if lang == "zh" else "Fit Results",
            "残差" if lang == "zh" else "Residuals",
            "归一化残差" if lang == "zh" else "Normalized Residuals",
        ),
        specs=[
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
        ],
    )

    # 添加数据点和误差棒
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            error_y=dict(
                type="data",
                array=ysigma,
                visible=True,
            ),
            mode="markers",
            name="数据" if lang == "zh" else "Data",
            marker=dict(color="blue", size=5),
        ),
        row=1,
        col=1,
    )

    # 添加拟合曲线
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prediction,
            mode="lines",
            name="拟合" if lang == "zh" else "Fit",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )

    # 添加各模板贡献
    for i, template in enumerate(templates):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=map_params[i] * template,
                mode="lines",
                name=f"模板 {i+1}" if lang == "zh" else f"Template {i+1}",
                line=dict(dash="dash", width=1),
            ),
            row=1,
            col=1,
        )

    # 添加残差
    fig.add_trace(
        go.Scatter(
            x=x,
            y=residuals,
            mode="markers",
            name="残差" if lang == "zh" else "Residuals",
            marker=dict(color="blue", size=5),
        ),
        row=2,
        col=1,
    )

    # 添加零线
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.zeros_like(x),
            mode="lines",
            name="零线" if lang == "zh" else "Zero Line",
            line=dict(color="red", width=1),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 添加归一化残差
    fig.add_trace(
        go.Scatter(
            x=x,
            y=normalized_residuals,
            mode="markers",
            name="归一化残差" if lang == "zh" else "Normalized Residuals",
            marker=dict(color="blue", size=5),
        ),
        row=3,
        col=1,
    )

    # 添加零线和±1线
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.zeros_like(x),
            mode="lines",
            name="零线" if lang == "zh" else "Zero Line",
            line=dict(color="red", width=1),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.ones_like(x),
            mode="lines",
            name="+1σ线" if lang == "zh" else "+1σ Line",
            line=dict(color="red", width=1, dash="dash"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=-np.ones_like(x),
            mode="lines",
            name="-1σ线" if lang == "zh" else "-1σ Line",
            line=dict(color="red", width=1, dash="dash"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # 更新布局
    chi2 = fit_stats["chi_square"]
    reduced_chi2 = fit_stats["reduced_chi_square"]
    dof = fit_stats["dof"]

    if lang == "zh":
        stats_text = f"<b>拟合统计:</b><br>χ² = {chi2:.2f}<br>简化χ² = {reduced_chi2:.3f}<br>自由度 = {dof}"
        x_label = "通道"
        y_labels = ["计数", "残差", "归一化残差"]
        title = "拟合结果与残差分析"
    else:
        stats_text = f"<b>Fit Statistics:</b><br>χ² = {chi2:.2f}<br>Reduced χ² = {reduced_chi2:.3f}<br>DOF = {dof}"
        x_label = "Channel"
        y_labels = ["Counts", "Residuals", "Normalized Residuals"]
        title = "Fit Results and Residual Analysis"

    fig.update_layout(
        height=900,
        title=title,
        annotations=[
            dict(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                align="left",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
            )
        ],
    )

    # 更新x轴和y轴标签
    fig.update_xaxes(title_text=x_label, row=3, col=1)
    for i, y_label in enumerate(y_labels):
        fig.update_yaxes(title_text=y_label, row=i + 1, col=1)

    # 保存为HTML文件
    fig.write_html(output_path)


def create_interactive_trace(results, output_path, lang="zh"):
    """
    创建交互式MCMC链迹图

    参数:
        results (dict): MCMC分析结果
        output_path (str): 输出文件路径
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)
    """
    if not PLOTLY_AVAILABLE:
        print("警告: Plotly不可用，跳过交互式图表创建")
        return

    samples = results["samples"]
    ndim = samples.shape[1]

    # 创建子图
    if lang == "zh":
        title = "MCMC链迹图"
        labels = [f"参数 {i+1}" for i in range(ndim)]
    else:
        title = "MCMC Chain Traces"
        labels = [f"Parameter {i+1}" for i in range(ndim)]

    fig = make_subplots(
        rows=ndim,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=labels,
    )

    # 添加链迹
    for i in range(ndim):
        fig.add_trace(
            go.Scatter(
                y=samples[:, i],
                mode="lines",
                name=labels[i],
                line=dict(color="black", width=1),
            ),
            row=i + 1,
            col=1,
        )

        # 添加中位数和置信区间
        median = results["medians"][i]
        lower = results["lower_bounds"][i]
        upper = results["upper_bounds"][i]

        fig.add_hline(
            y=median,
            line=dict(color="red", width=2),
            row=i + 1,
            col=1,
        )

        fig.add_hline(
            y=lower,
            line=dict(color="red", width=1.5, dash="dash"),
            row=i + 1,
            col=1,
        )

        fig.add_hline(
            y=upper,
            line=dict(color="red", width=1.5, dash="dash"),
            row=i + 1,
            col=1,
        )

    # 更新布局
    fig.update_layout(
        height=200 * ndim,
        title=title,
        showlegend=False,
    )

    # 更新x轴标签
    if lang == "zh":
        fig.update_xaxes(title_text="步数", row=ndim, col=1)
    else:
        fig.update_xaxes(title_text="Step", row=ndim, col=1)

    # 保存为HTML文件
    fig.write_html(output_path)
