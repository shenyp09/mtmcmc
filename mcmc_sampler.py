#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCMC采样模块
----------
使用emcee库实现MCMC采样
"""

import numpy as np
import emcee
from multiprocessing import Pool

from model import log_probability


def _log_prob_wrapper(
    params, y, ysigma, templates, templates_sigma, priors, default_prior
):
    """
    对数后验概率函数包装器，用于emcee采样器

    参数:
        params: 模型参数
        y, ysigma, templates, templates_sigma: 数据和模板
        priors: 先验分布列表
        default_prior: 默认先验分布

    返回:
        float: 对数后验概率
    """
    return log_probability(
        params,
        y,
        ysigma,
        templates,
        templates_sigma,
        priors=priors,
        default_prior=default_prior,
    )


def run_mcmc(
    y,
    ysigma,
    templates,
    templates_sigma,
    ndim=None,
    nwalkers=32,
    nsteps=5000,
    burnin=1000,
    priors=None,
    default_prior=None,
    progress=False,
    ncores=None,
):
    """
    运行MCMC采样

    参数:
        y (array): 目标能谱数据
        ysigma (array): 目标能谱标准差
        templates (list): 模板能谱列表
        templates_sigma (list): 模板能谱标准差列表
        ndim (int): 参数维度
        nwalkers (int): MCMC walkers数量
        nsteps (int): MCMC采样步数
        burnin (int): MCMC预热步数
        priors (list): 先验分布列表
        default_prior (dict): 默认先验分布
        progress (bool): 是否显示进度条
        ncores (int): 使用的CPU核心数

    返回:
        tuple: (sampler, pos, prob, state)
            - sampler: emcee.EnsembleSampler对象
            - pos: 最后一步的采样位置
            - prob: 最后一步的对数概率
            - state: 最后一步的随机状态
    """
    # 如果未指定参数维度，使用模板数量
    if ndim is None:
        ndim = len(templates)

    # 初始化参数位置
    # 通过简单的线性回归获得初始参数猜测值
    initial_guess = initial_param_guess(y, templates)

    # 在初始猜测值附近生成随机初始点
    pos = initial_guess + 0.01 * np.random.randn(nwalkers, ndim)
    # 确保所有参数都是正数
    pos = np.abs(pos)

    moves = [(emcee.moves.DESnookerMove(), 0.8), (emcee.moves.DEMove(), 0.2)]

    # 设置并行计算
    with Pool(processes=ncores) as pool:
        # 创建采样器
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            _log_prob_wrapper,
            moves=moves,
            args=(y, ysigma, templates, templates_sigma, priors, default_prior),
            pool=pool,
        )

        # 执行burn-in阶段
        if burnin > 0:
            if progress:
                print(f"执行burn-in阶段 ({burnin}步)...")
            pos, prob, rstate = sampler.run_mcmc(pos, burnin, progress=progress)
            sampler.reset()
        else:
            rstate = None

        # 执行正式采样
        if progress:
            print(f"执行MCMC采样 ({nsteps}步)...")
        pos, prob, rstate = sampler.run_mcmc(
            pos, nsteps, rstate0=rstate, progress=progress
        )

    return sampler, pos, prob, rstate


def initial_param_guess(y, templates):
    """
    使用非负最小二乘法获取参数的初始猜测值

    参数:
        y (array): 目标能谱数据
        templates (list): 模板能谱列表

    返回:
        array: 参数初始猜测值
    """
    from scipy.optimize import nnls

    # 构建设计矩阵
    A = np.column_stack(templates)

    # 使用非负最小二乘求解
    params, _ = nnls(A, y)

    # 确保所有参数为正
    params = np.maximum(params, 1e-5)

    return params


def check_convergence(sampler, burnin=0, threshold=50):
    """
    检查MCMC链的收敛性

    参数:
        sampler (emcee.EnsembleSampler): MCMC采样器
        burnin (int): 预热步数
        threshold (int): 自相关时间的阈值

    返回:
        tuple: (converged, tau)
            - converged (bool): 是否收敛
            - tau (array): 自相关时间
    """
    # 计算自相关时间
    try:
        tau = sampler.get_autocorr_time()
        converged = np.all(tau * threshold < sampler.iteration)
        return converged, tau
    except emcee.autocorr.AutocorrError:
        # 如果自相关时间计算失败，认为未收敛
        return False, None


def get_flat_samples(sampler, burnin=0, thin=1):
    """
    获取展平的MCMC样本

    参数:
        sampler (emcee.EnsembleSampler): MCMC采样器
        burnin (int): 预热步数
        thin (int): 抽稀间隔

    返回:
        array: 展平的样本数组
    """
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    return samples
