#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HTML报告生成模块
--------------
生成包含分析结果的HTML报告
"""

import os
import numpy as np
import base64
import io
from pathlib import Path
import jinja2
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

from model import model_predict
from analyzer import error_contribution_analysis
from visualizer import create_plots


def generate_html_report(
    y,
    ysigma,
    templates,
    templates_sigma,
    results,
    fit_stats,
    figure_files,
    template_dir=None,
    output_dir="results",
    interactive=True,
    languages=["zh"],
):
    """
    生成HTML报告

    参数:
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表
        results (dict): MCMC分析结果
        fit_stats (dict): 拟合统计量
        figure_files (dict): 图表文件路径字典
        template_dir (str): HTML模板目录，如果为None则使用内置模板
        output_dir (str): 输出目录
        interactive (bool): 是否包含交互式图表
        languages (list): 要生成的报告语言列表，可选值：["zh", "en"]或["zh"]或["en"]

    返回:
        dict: 各语言HTML报告文件路径
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 支持的语言列表
    supported_languages = ["zh", "en"]
    # 过滤掉不支持的语言
    languages = [lang for lang in languages if lang in supported_languages]
    # 如果没有指定有效语言，默认使用中文
    if not languages:
        languages = ["zh"]

    output_files = {}

    for lang in languages:
        # 为每种语言创建子目录
        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        # 为每种语言生成对应的图表
        lang_figure_files = create_plots(
            y,
            ysigma,
            templates,
            templates_sigma,
            results,
            fit_stats,
            output_dir=lang_dir,
            interactive=interactive,
            lang=lang,
        )

        # 生成报告数据
        report_data = prepare_report_data(
            y,
            ysigma,
            templates,
            templates_sigma,
            results,
            fit_stats,
            lang_figure_files,
            interactive,
            lang,
        )

        # 获取HTML模板
        template_text = get_html_template(template_dir, lang)

        # 渲染HTML
        html_content = render_html(template_text, report_data)

        # 写入文件
        output_file = os.path.join(lang_dir, "mtmcmc_report.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        output_files[lang] = output_file

    return output_files


def prepare_report_data(
    y,
    ysigma,
    templates,
    templates_sigma,
    results,
    fit_stats,
    figure_files,
    interactive,
    lang="zh",
):
    """
    准备报告数据

    参数:
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表
        results (dict): MCMC分析结果
        fit_stats (dict): 拟合统计量
        figure_files (dict): 图表文件路径字典
        interactive (bool): 是否包含交互式图表
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)

    返回:
        dict: 报告数据字典
    """
    # 获取参数信息
    params_info = []
    for i in range(len(results["medians"])):
        params_info.append(
            {
                "index": i + 1,
                "median": results["medians"][i],
                "lower": results["lower_bounds"][i],
                "upper": results["upper_bounds"][i],
                "mean": results["means"][i],
                "std": results["stds"][i],
                "map": results["map_estimate"][i],
            }
        )

    # 分析拟合优度并按语言设置质量文本
    rchisq = fit_stats["reduced_chi_square"]
    if lang == "zh":
        quality_text = (
            "优秀"
            if 0.8 <= rchisq <= 1.2
            else "一般" if 0.5 <= rchisq <= 1.5 else "较差"
        )
    else:  # English
        quality_text = (
            "Excellent"
            if 0.8 <= rchisq <= 1.2
            else "Fair" if 0.5 <= rchisq <= 1.5 else "Poor"
        )

    # 准备拟合数据表格
    fit_data_table = []
    x = np.arange(len(y))
    prediction = fit_stats["prediction"]
    residuals = fit_stats["residuals"]
    normalized_residuals = fit_stats["normalized_residuals"]
    total_sigma = np.sqrt(fit_stats["residuals"] ** 2 + prediction**2)

    # 限制表格长度，避免过大
    max_rows = 100
    if len(y) > max_rows:
        step = len(y) // max_rows
        indices = np.arange(0, len(y), step)
    else:
        indices = np.arange(len(y))

    for i in indices:
        fit_data_table.append(
            {
                "channel": i,
                "data": y[i],
                "error": ysigma[i],
                "prediction": prediction[i],
                "residual": residuals[i],
                "norm_residual": normalized_residuals[i],
            }
        )

    # 准备模板贡献数据
    template_contributions = fit_stats["template_contributions"]

    # 准备误差贡献分析
    error_analysis = error_contribution_analysis(
        results["map_estimate"], ysigma, templates_sigma
    )

    # 检查收敛性
    convergence_info = results["convergence"]
    converged = convergence_info["converged"]

    # 根据语言设置收敛性状态文本
    if lang == "zh":
        status_text = "良好" if converged else "不足"
        recommendation_text = "结果可靠" if converged else "建议增加采样步数"
        unknown_text = "未知"
        na_text = "无法计算"
        increase_steps_text = "建议增加采样步数以获得可靠结果"
    else:  # English
        status_text = "Good" if converged else "Insufficient"
        recommendation_text = (
            "Results are reliable" if converged else "Increase sampling steps"
        )
        unknown_text = "Unknown"
        na_text = "Not calculable"
        increase_steps_text = "Increase sampling steps for reliable results"

    if convergence_info["autocorr_time"] is not None:
        autocorr_text = ", ".join(
            [f"{t:.1f}" for t in convergence_info["autocorr_time"]]
        )
        threshold_ratio = convergence_info["threshold_ratio"]

        convergence_status = {
            "converged": converged,
            "status": status_text,
            "autocorr_text": autocorr_text,
            "threshold_ratio": threshold_ratio,
            "recommendation": recommendation_text,
        }
    else:
        convergence_status = {
            "converged": False,
            "status": unknown_text,
            "autocorr_text": na_text,
            "threshold_ratio": "N/A",
            "recommendation": increase_steps_text,
        }

    # 准备互动式图表链接
    interactive_plots = {}
    if interactive:
        for key in ["interactive_corner", "interactive_fit", "interactive_trace"]:
            if key in figure_files:
                interactive_plots[key] = os.path.basename(figure_files[key])

    # 根据语言设置报告标题
    if lang == "zh":
        title = "Multi-Template MCMC贝叶斯分析报告"
    else:  # English
        title = "Multi-Template MCMC Bayesian Analysis Report"

    # 创建报告数据字典
    report_data = {
        "title": title,
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_info": {"data_length": len(y), "num_templates": len(templates)},
        "params_info": params_info,
        "fit_stats": {
            "chi_square": fit_stats["chi_square"],
            "reduced_chi_square": fit_stats["reduced_chi_square"],
            "dof": fit_stats["dof"],
            "p_value": fit_stats["p_value"],
            "bic": fit_stats["bic"],
            "aic": fit_stats["aic"],
            "quality": quality_text,
        },
        "template_contributions": template_contributions,
        "error_analysis": {
            "target_contribution": error_analysis["target_contribution"],
            "template_contributions": error_analysis["template_contributions"],
        },
        "convergence": convergence_status,
        "fit_data_table": fit_data_table,
        "figures": {
            key: os.path.basename(path)
            for key, path in figure_files.items()
            if not key.startswith("interactive_")
        },
        "interactive_plots": interactive_plots,
        "lang": lang,  # 添加语言信息到报告数据
    }

    return report_data


def get_html_template(template_dir=None, lang="zh"):
    """
    获取HTML模板

    参数:
        template_dir (str): HTML模板目录，如果为None则使用内置模板
        lang (str): 语言，可选值："zh"(中文)或"en"(英文)

    返回:
        str: HTML模板文本
    """
    if template_dir is not None:
        # 从指定目录加载模板
        template_filename = (
            f"report_template_{lang}.html" if lang == "en" else "report_template.html"
        )
        template_path = os.path.join(template_dir, template_filename)
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()

    # 使用内置模板
    if lang == "en":
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            padding-top: 20px;
        }
        .container { max-width: 1200px; }
        .header { margin-bottom: 30px; }
        .section { margin-bottom: 40px; }
        .figure { margin: 20px 0; text-align: center; }
        .figure img { max-width: 100%; height: auto; }
        .table-responsive { margin: 20px 0; }
        .fit-stats { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
        .param-box { 
            background-color: #e9ecef; 
            padding: 10px; 
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .convergence-good { color: green; }
        .convergence-bad { color: red; }
        .nav-pills .nav-link.active { background-color: #0d6efd; }
        .tab-content { padding: 20px 0; }
        .footer { margin-top: 50px; padding: 20px 0; border-top: 1px solid #dee2e6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="text-center">{{ title }}</h1>
            <p class="text-center text-muted">Generated on: {{ generation_time }}</p>
        </div>
        
        <nav>
            <div class="nav nav-pills mb-3" id="nav-tab" role="tablist">
                <button class="nav-link active" id="nav-summary-tab" data-bs-toggle="tab" data-bs-target="#nav-summary" type="button" role="tab" aria-controls="nav-summary" aria-selected="true">Summary</button>
                <button class="nav-link" id="nav-posterior-tab" data-bs-toggle="tab" data-bs-target="#nav-posterior" type="button" role="tab" aria-controls="nav-posterior" aria-selected="false">Posterior Distribution</button>
                <button class="nav-link" id="nav-fit-tab" data-bs-toggle="tab" data-bs-target="#nav-fit" type="button" role="tab" aria-controls="nav-fit" aria-selected="false">Fit Results</button>
                <button class="nav-link" id="nav-error-tab" data-bs-toggle="tab" data-bs-target="#nav-error" type="button" role="tab" aria-controls="nav-error" aria-selected="false">Error Analysis</button>
                <button class="nav-link" id="nav-data-tab" data-bs-toggle="tab" data-bs-target="#nav-data" type="button" role="tab" aria-controls="nav-data" aria-selected="false">Data Table</button>
                {% if interactive_plots %}
                <button class="nav-link" id="nav-interactive-tab" data-bs-toggle="tab" data-bs-target="#nav-interactive" type="button" role="tab" aria-controls="nav-interactive" aria-selected="false">Interactive Plots</button>
                {% endif %}
            </div>
        </nav>
        
        <div class="tab-content" id="nav-tabContent">
            <!-- Summary Tab -->
            <div class="tab-pane fade show active" id="nav-summary" role="tabpanel" aria-labelledby="nav-summary-tab">
                <div class="section">
                    <h2>Analysis Summary</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="fit-stats">
                                <h4>Data Information</h4>
                                <ul>
                                    <li>Spectrum Data Length: {{ data_info.data_length }}</li>
                                    <li>Number of Templates: {{ data_info.num_templates }}</li>
                                </ul>
                                
                                <h4>Fit Statistics</h4>
                                <ul>
                                    <li>Chi-square: {{ fit_stats.chi_square|round(2) }}</li>
                                    <li>Reduced Chi-square: {{ fit_stats.reduced_chi_square|round(3) }}</li>
                                    <li>Degrees of Freedom: {{ fit_stats.dof }}</li>
                                    <li>p-value: {{ fit_stats.p_value|round(4) }}</li>
                                    <li>BIC: {{ fit_stats.bic|round(2) }}</li>
                                    <li>AIC: {{ fit_stats.aic|round(2) }}</li>
                                    <li>Fit Quality: {{ fit_stats.quality }}</li>
                                </ul>
                                
                                <h4>Convergence</h4>
                                <ul>
                                    <li>Convergence Status: <span class="{% if convergence.converged %}convergence-good{% else %}convergence-bad{% endif %}">{{ convergence.status }}</span></li>
                                    <li>Autocorrelation Time: {{ convergence.autocorr_text }}</li>
                                    <li>Recommendation: {{ convergence.recommendation }}</li>
                                </ul>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h4>Best Fit Parameters</h4>
                            <div class="row">
                                {% for param in params_info %}
                                <div class="col-md-6">
                                    <div class="param-box">
                                        <h5>Parameter {{ param.index }}</h5>
                                        <p>MAP Estimate: {{ param.map|round(4) }}</p>
                                        <p>Median: {{ param.median|round(4) }}</p>
                                        <p>68% Confidence Interval: [{{ param.lower|round(4) }}, {{ param.upper|round(4) }}]</p>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                            
                            <h4>Template Contribution Ratio</h4>
                            <div class="figure">
                                <img src="{{ figures.contribution }}" alt="Template Contribution Pie Chart" class="img-fluid">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Posterior Distribution Tab -->
            <div class="tab-pane fade" id="nav-posterior" role="tabpanel" aria-labelledby="nav-posterior-tab">
                <div class="section">
                    <h2>Parameter Posterior Distribution</h2>
                    <div class="figure">
                        <img src="{{ figures.corner }}" alt="Parameter Posterior Distribution Corner Plot" class="img-fluid">
                        <p class="text-muted">Parameter posterior distribution corner plot, showing marginal distributions on the diagonal and joint distributions off-diagonal</p>
                    </div>
                    
                    <h2>MCMC Chain Traces</h2>
                    <div class="figure">
                        <img src="{{ figures.trace }}" alt="MCMC Chain Traces" class="img-fluid">
                        <p class="text-muted">MCMC chain traces, with red solid lines showing parameter medians and red dashed lines showing 68% confidence intervals</p>
                    </div>
                </div>
            </div>
            
            <!-- Fit Results Tab -->
            <div class="tab-pane fade" id="nav-fit" role="tabpanel" aria-labelledby="nav-fit-tab">
                <div class="section">
                    <h2>Fit Results</h2>
                    <div class="figure">
                        <img src="{{ figures.fit }}" alt="Fit Results Plot" class="img-fluid">
                        <p class="text-muted">Fit results plot, with blue points representing data, red line showing the fit result, and dashed lines showing individual template contributions</p>
                    </div>
                    
                    <h2>Residual Analysis</h2>
                    <div class="figure">
                        <img src="{{ figures.residual }}" alt="Residual Plot" class="img-fluid">
                        <p class="text-muted">The upper plot shows raw residuals, the lower plot shows normalized residuals (residuals/σ)</p>
                    </div>
                </div>
            </div>
            
            <!-- Error Analysis Tab -->
            <div class="tab-pane fade" id="nav-error" role="tabpanel" aria-labelledby="nav-error-tab">
                <div class="section">
                    <h2>Error Contribution Analysis</h2>
                    <div class="figure">
                        <img src="{{ figures.error }}" alt="Error Contribution Analysis Plot" class="img-fluid">
                        <p class="text-muted">Average percentage contribution of each error source to the total error</p>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Target Spectrum Error Contribution</h4>
                            <ul>
                                <li>Average Contribution: {{ error_analysis.target_contribution.mean|round(2) }}%</li>
                                <li>Maximum Contribution: {{ error_analysis.target_contribution.max|round(2) }}%</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h4>Template Spectra Error Contribution</h4>
                            <ul>
                                {% for tc in error_analysis.template_contributions %}
                                <li>Template {{ tc.index+1 }} (Weight: {{ tc.weight|round(4) }}): Average {{ tc.mean_contribution|round(2) }}%, Maximum {{ tc.max_contribution|round(2) }}%</li>
                                {% endfor %}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Data Table Tab -->
            <div class="tab-pane fade" id="nav-data" role="tabpanel" aria-labelledby="nav-data-tab">
                <div class="section">
                    <h2>Data Table</h2>
                    <p>Showing partial data points (maximum 100 rows)</p>
                    <div class="table-responsive">
                        <table class="table table-striped table-sm">
                            <thead>
                                <tr>
                                    <th>Channel</th>
                                    <th>Data</th>
                                    <th>Error</th>
                                    <th>Prediction</th>
                                    <th>Residual</th>
                                    <th>Normalized Residual</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for row in fit_data_table %}
                                <tr>
                                    <td>{{ row.channel }}</td>
                                    <td>{{ row.data|round(2) }}</td>
                                    <td>{{ row.error|round(2) }}</td>
                                    <td>{{ row.prediction|round(2) }}</td>
                                    <td>{{ row.residual|round(2) }}</td>
                                    <td>{{ row.norm_residual|round(2) }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="mt-4">
                        <h4>Template Contribution Details</h4>
                        <div class="table-responsive">
                            <table class="table table-striped table-sm">
                                <thead>
                                    <tr>
                                        <th>Template</th>
                                        <th>Weight</th>
                                        <th>Total Counts</th>
                                        <th>Contribution Percentage</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for tc in template_contributions %}
                                    <tr>
                                        <td>Template {{ tc.index+1 }}</td>
                                        <td>{{ tc.weight|round(4) }}</td>
                                        <td>{{ tc.counts|round(2) }}</td>
                                        <td>{{ tc.percentage|round(2) }}%</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            {% if interactive_plots %}
            <!-- Interactive Plots Tab -->
            <div class="tab-pane fade" id="nav-interactive" role="tabpanel" aria-labelledby="nav-interactive-tab">
                <div class="section">
                    <h2>Interactive Plots</h2>
                    <div class="list-group">
                        {% if interactive_plots.interactive_fit %}
                        <a href="{{ interactive_plots.interactive_fit }}" class="list-group-item list-group-item-action" target="_blank">
                            Interactive Fit Results Plot
                        </a>
                        {% endif %}
                        
                        {% if interactive_plots.interactive_corner %}
                        <a href="{{ interactive_plots.interactive_corner }}" class="list-group-item list-group-item-action" target="_blank">
                            Interactive Parameter Posterior Distribution Plot
                        </a>
                        {% endif %}
                        
                        {% if interactive_plots.interactive_trace %}
                        <a href="{{ interactive_plots.interactive_trace }}" class="list-group-item list-group-item-action" target="_blank">
                            Interactive MCMC Chain Traces Plot
                        </a>
                        {% endif %}
                    </div>
                    <p class="mt-3 text-muted">Click the links to open interactive plots in a new tab</p>
                </div>
            </div>
            {% endif %}
            
        </div>
        
        <div class="footer text-center">
            <p>Multi Template MCMC Bayesian Analysis | Generated on {{ generation_time }}</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    else:  # 中文模板
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { 
            font-family: "SimHei", Arial, sans-serif; 
            padding-top: 20px;
        }
        .container { max-width: 1200px; }
        .header { margin-bottom: 30px; }
        .section { margin-bottom: 40px; }
        .figure { margin: 20px 0; text-align: center; }
        .figure img { max-width: 100%; height: auto; }
        .table-responsive { margin: 20px 0; }
        .fit-stats { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
        .param-box { 
            background-color: #e9ecef; 
            padding: 10px; 
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .convergence-good { color: green; }
        .convergence-bad { color: red; }
        .nav-pills .nav-link.active { background-color: #0d6efd; }
        .tab-content { padding: 20px 0; }
        .footer { margin-top: 50px; padding: 20px 0; border-top: 1px solid #dee2e6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="text-center">{{ title }}</h1>
            <p class="text-center text-muted">生成时间: {{ generation_time }}</p>
        </div>
        
        <nav>
            <div class="nav nav-pills mb-3" id="nav-tab" role="tablist">
                <button class="nav-link active" id="nav-summary-tab" data-bs-toggle="tab" data-bs-target="#nav-summary" type="button" role="tab" aria-controls="nav-summary" aria-selected="true">摘要</button>
                <button class="nav-link" id="nav-posterior-tab" data-bs-toggle="tab" data-bs-target="#nav-posterior" type="button" role="tab" aria-controls="nav-posterior" aria-selected="false">后验分布</button>
                <button class="nav-link" id="nav-fit-tab" data-bs-toggle="tab" data-bs-target="#nav-fit" type="button" role="tab" aria-controls="nav-fit" aria-selected="false">拟合结果</button>
                <button class="nav-link" id="nav-error-tab" data-bs-toggle="tab" data-bs-target="#nav-error" type="button" role="tab" aria-controls="nav-error" aria-selected="false">误差分析</button>
                <button class="nav-link" id="nav-data-tab" data-bs-toggle="tab" data-bs-target="#nav-data" type="button" role="tab" aria-controls="nav-data" aria-selected="false">数据表</button>
                {% if interactive_plots %}
                <button class="nav-link" id="nav-interactive-tab" data-bs-toggle="tab" data-bs-target="#nav-interactive" type="button" role="tab" aria-controls="nav-interactive" aria-selected="false">交互图表</button>
                {% endif %}
            </div>
        </nav>
        
        <div class="tab-content" id="nav-tabContent">
            <!-- 摘要标签页 -->
            <div class="tab-pane fade show active" id="nav-summary" role="tabpanel" aria-labelledby="nav-summary-tab">
                <div class="section">
                    <h2>分析摘要</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="fit-stats">
                                <h4>数据信息</h4>
                                <ul>
                                    <li>能谱数据长度: {{ data_info.data_length }}</li>
                                    <li>模板数量: {{ data_info.num_templates }}</li>
                                </ul>
                                
                                <h4>拟合统计</h4>
                                <ul>
                                    <li>卡方值: {{ fit_stats.chi_square|round(2) }}</li>
                                    <li>简化卡方: {{ fit_stats.reduced_chi_square|round(3) }}</li>
                                    <li>自由度: {{ fit_stats.dof }}</li>
                                    <li>p值: {{ fit_stats.p_value|round(4) }}</li>
                                    <li>BIC: {{ fit_stats.bic|round(2) }}</li>
                                    <li>AIC: {{ fit_stats.aic|round(2) }}</li>
                                    <li>拟合质量: {{ fit_stats.quality }}</li>
                                </ul>
                                
                                <h4>收敛性</h4>
                                <ul>
                                    <li>收敛状态: <span class="{% if convergence.converged %}convergence-good{% else %}convergence-bad{% endif %}">{{ convergence.status }}</span></li>
                                    <li>自相关时间: {{ convergence.autocorr_text }}</li>
                                    <li>建议: {{ convergence.recommendation }}</li>
                                </ul>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h4>最佳拟合参数</h4>
                            <div class="row">
                                {% for param in params_info %}
                                <div class="col-md-6">
                                    <div class="param-box">
                                        <h5>参数 {{ param.index }}</h5>
                                        <p>MAP估计: {{ param.map|round(4) }}</p>
                                        <p>中位数: {{ param.median|round(4) }}</p>
                                        <p>68%置信区间: [{{ param.lower|round(4) }}, {{ param.upper|round(4) }}]</p>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                            
                            <h4>模板贡献比例</h4>
                            <div class="figure">
                                <img src="{{ figures.contribution }}" alt="模板贡献饼图" class="img-fluid">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 后验分布标签页 -->
            <div class="tab-pane fade" id="nav-posterior" role="tabpanel" aria-labelledby="nav-posterior-tab">
                <div class="section">
                    <h2>参数后验分布</h2>
                    <div class="figure">
                        <img src="{{ figures.corner }}" alt="参数后验分布角图" class="img-fluid">
                        <p class="text-muted">参数后验分布角图，对角线上是单参数边缘分布，非对角线是参数联合分布</p>
                    </div>
                    
                    <h2>MCMC链迹图</h2>
                    <div class="figure">
                        <img src="{{ figures.trace }}" alt="MCMC链迹图" class="img-fluid">
                        <p class="text-muted">MCMC链迹图，红色实线是参数中位数，红色虚线是68%置信区间</p>
                    </div>
                </div>
            </div>
            
            <!-- 拟合结果标签页 -->
            <div class="tab-pane fade" id="nav-fit" role="tabpanel" aria-labelledby="nav-fit-tab">
                <div class="section">
                    <h2>拟合结果</h2>
                    <div class="figure">
                        <img src="{{ figures.fit }}" alt="拟合结果图" class="img-fluid">
                        <p class="text-muted">拟合结果图，蓝色点是数据，红色线是拟合结果，虚线是各模板贡献</p>
                    </div>
                    
                    <h2>残差分析</h2>
                    <div class="figure">
                        <img src="{{ figures.residual }}" alt="残差图" class="img-fluid">
                        <p class="text-muted">上图为原始残差，下图为归一化残差(残差/σ)</p>
                    </div>
                </div>
            </div>
            
            <!-- 误差分析标签页 -->
            <div class="tab-pane fade" id="nav-error" role="tabpanel" aria-labelledby="nav-error-tab">
                <div class="section">
                    <h2>误差贡献分析</h2>
                    <div class="figure">
                        <img src="{{ figures.error }}" alt="误差贡献分析图" class="img-fluid">
                        <p class="text-muted">各误差源对总误差的平均贡献百分比</p>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <h4>目标谱误差贡献</h4>
                            <ul>
                                <li>平均贡献: {{ error_analysis.target_contribution.mean|round(2) }}%</li>
                                <li>最大贡献: {{ error_analysis.target_contribution.max|round(2) }}%</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h4>模板谱误差贡献</h4>
                            <ul>
                                {% for tc in error_analysis.template_contributions %}
                                <li>模板 {{ tc.index+1 }} (权重: {{ tc.weight|round(4) }}): 平均 {{ tc.mean_contribution|round(2) }}%, 最大 {{ tc.max_contribution|round(2) }}%</li>
                                {% endfor %}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 数据表标签页 -->
            <div class="tab-pane fade" id="nav-data" role="tabpanel" aria-labelledby="nav-data-tab">
                <div class="section">
                    <h2>数据表</h2>
                    <p>显示部分数据点（最多100行）</p>
                    <div class="table-responsive">
                        <table class="table table-striped table-sm">
                            <thead>
                                <tr>
                                    <th>通道</th>
                                    <th>数据</th>
                                    <th>误差</th>
                                    <th>预测</th>
                                    <th>残差</th>
                                    <th>归一化残差</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for row in fit_data_table %}
                                <tr>
                                    <td>{{ row.channel }}</td>
                                    <td>{{ row.data|round(2) }}</td>
                                    <td>{{ row.error|round(2) }}</td>
                                    <td>{{ row.prediction|round(2) }}</td>
                                    <td>{{ row.residual|round(2) }}</td>
                                    <td>{{ row.norm_residual|round(2) }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="mt-4">
                        <h4>模板贡献详情</h4>
                        <div class="table-responsive">
                            <table class="table table-striped table-sm">
                                <thead>
                                    <tr>
                                        <th>模板</th>
                                        <th>权重</th>
                                        <th>总计数</th>
                                        <th>贡献百分比</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for tc in template_contributions %}
                                    <tr>
                                        <td>模板 {{ tc.index+1 }}</td>
                                        <td>{{ tc.weight|round(4) }}</td>
                                        <td>{{ tc.counts|round(2) }}</td>
                                        <td>{{ tc.percentage|round(2) }}%</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            {% if interactive_plots %}
            <!-- 交互图表标签页 -->
            <div class="tab-pane fade" id="nav-interactive" role="tabpanel" aria-labelledby="nav-interactive-tab">
                <div class="section">
                    <h2>交互式图表</h2>
                    <div class="list-group">
                        {% if interactive_plots.interactive_fit %}
                        <a href="{{ interactive_plots.interactive_fit }}" class="list-group-item list-group-item-action" target="_blank">
                            交互式拟合结果图
                        </a>
                        {% endif %}
                        
                        {% if interactive_plots.interactive_corner %}
                        <a href="{{ interactive_plots.interactive_corner }}" class="list-group-item list-group-item-action" target="_blank">
                            交互式参数后验分布图
                        </a>
                        {% endif %}
                        
                        {% if interactive_plots.interactive_trace %}
                        <a href="{{ interactive_plots.interactive_trace }}" class="list-group-item list-group-item-action" target="_blank">
                            交互式MCMC链迹图
                        </a>
                        {% endif %}
                    </div>
                    <p class="mt-3 text-muted">点击链接在新标签页中打开交互式图表</p>
                </div>
            </div>
            {% endif %}
            
        </div>
        
        <div class="footer text-center">
            <p>Multi Template MCMC贝叶斯分析 | 生成于 {{ generation_time }}</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


def render_html(template_text, report_data):
    """渲染HTML模板"""
    template = jinja2.Template(template_text)
    html_content = template.render(**report_data)
    return html_content
