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

数据加载模块
---------
用于加载和预处理能谱数据
"""

import numpy as np


def load_data(filename):
    """
    从文件加载能谱数据

    参数:
        filename (str): 能谱数据文件路径

    返回:
        dict: 包含以下键的字典:
            - energy: 能量值数组
            - counts: 计数值数组
            - errors: 标准差数组
    """
    try:
        data = np.loadtxt(filename, delimiter=None)

        # 检查数据列数
        if data.shape[1] < 3:
            raise ValueError(
                f"数据文件{filename}格式不正确，至少需要3列: 能量、计数、标准差"
            )

        # 提取列
        energy = data[:, 0]
        counts = data[:, 1]
        errors = data[:, 2]

        # 基本检查
        if np.any(errors <= 0):
            print(f"警告: 文件{filename}中存在零或负误差，将用数据的平方根替代")
            bad_indices = errors <= 0
            errors[bad_indices] = np.sqrt(np.abs(counts[bad_indices]) + 1e-10)

        return {
            "energy": energy,
            "counts": counts,
            "errors": errors,
            "filename": filename,
        }

    except Exception as e:
        raise IOError(f"无法读取文件{filename}: {str(e)}")


def preprocess_data(target_data, template_data_list, error_handling="both"):
    """
    预处理目标能谱和模板能谱数据

    参数:
        target_data (dict): 目标能谱数据
        template_data_list (list): 模板能谱数据列表
        error_handling (str): 误差处理方式 {'target', 'template', 'both'}

    返回:
        tuple: (y, ysigma, templates, templates_sigma)
            - y: 目标能谱计数值数组
            - ysigma: 目标能谱标准差数组
            - templates: 模板能谱计数值数组列表
            - templates_sigma: 模板能谱标准差数组列表
    """
    # 获取能量轴
    target_energy = target_data["energy"]

    # 确认所有模板具有相同的能量轴
    for i, template in enumerate(template_data_list):
        if len(template["energy"]) != len(target_energy):
            raise ValueError(f"模板 #{i+1} 长度与目标谱不匹配")

        if not np.allclose(template["energy"], target_energy):
            raise ValueError(
                f"模板 #{i+1} 能量轴与目标谱不匹配，请确保所有能谱具有相同的能量轴"
            )

    # 提取数据
    y = target_data["counts"]
    ysigma = target_data["errors"]

    templates = [template["counts"] for template in template_data_list]
    templates_sigma = [template["errors"] for template in template_data_list]

    # 根据误差处理方式调整
    if error_handling == "target":
        # 仅考虑目标谱误差
        templates_sigma = [np.zeros_like(sigma) for sigma in templates_sigma]
    elif error_handling == "template":
        # 仅考虑模板谱误差
        ysigma = np.zeros_like(ysigma)
    # 'both'情况下保持原样

    return y, ysigma, templates, templates_sigma


def generate_synthetic_data(n_points=4096, n_templates=3, weights=None, seed=42):
    """
    生成合成数据用于测试

    参数:
        n_points (int): 数据点数量
        n_templates (int): 模板数量
        weights (list): 预设的模板权重，如果为None则随机生成
        seed (int): 随机种子

    返回:
        tuple: (y, ysigma, templates, templates_sigma, true_weights)
    """
    np.random.seed(seed)

    # 生成能量轴
    energy = np.linspace(0, 100, n_points)

    # 生成模板
    templates = []
    templates_sigma = []

    for i in range(n_templates):
        # 每个模板是若干高斯峰的叠加
        template = np.zeros(n_points)
        n_peaks = np.random.randint(3, 8)

        for _ in range(n_peaks):
            center = np.random.uniform(10, 90)
            width = np.random.uniform(1, 5)
            amplitude = np.random.uniform(10, 100)
            template += amplitude * np.exp(-0.5 * ((energy - center) / width) ** 2)

        # 添加模板误差 (假设为泊松分布)
        template_sigma = np.sqrt(np.abs(template) + 1)

        templates.append(template)
        templates_sigma.append(template_sigma)

    # 生成真实权重并构建合成目标谱
    if weights is None:
        true_weights = np.random.uniform(0.5, 2.0, n_templates)
    else:
        # 确保权重数量与模板数量匹配
        if len(weights) != n_templates:
            print(
                f"警告: 提供的权重数量({len(weights)})与模板数量({n_templates})不匹配"
            )
            # 如果提供的权重不足，则随机生成剩余的权重
            if len(weights) < n_templates:
                additional_weights = np.random.uniform(
                    0.5, 2.0, n_templates - len(weights)
                )
                true_weights = np.concatenate([weights, additional_weights])
            else:
                # 如果提供的权重过多，则只使用前n_templates个
                true_weights = weights[:n_templates]
        else:
            true_weights = weights

    y = np.zeros(n_points)
    for i, template in enumerate(templates):
        y += true_weights[i] * template

    # 添加噪声
    ysigma = np.sqrt(np.abs(y) + 1)
    y += np.random.normal(0, ysigma)

    return y, ysigma, templates, templates_sigma, true_weights
