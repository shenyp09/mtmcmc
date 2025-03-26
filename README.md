# Multi Template MCMC贝叶斯分析

用于能谱拟合的多模板MCMC贝叶斯分析算法，支持考虑多种误差源和生成HTML报告。

## 功能特点

- 多模板能谱拟合：使用MCMC方法拟合由多个模板线性组合的能谱
- 综合误差处理：同时考虑目标能谱和模板能谱的测量误差
- 多种先验分布：支持为每个模板单独设置不同的先验分布（均匀分布、正态分布和对数正态分布）
- 统计分析：计算参数估计值、置信区间、拟合优度等统计量
- 可视化：生成各种静态图表和交互式图表
- HTML报告：自动生成包含分析结果的HTML报告

## 安装依赖

```bash
pip install numpy scipy matplotlib seaborn emcee corner jinja2 plotly
```

## 使用方法

### 基本用法

```bash
# 编辑config.py配置文件，设置所需参数
python mtmcmc.py
```

### 配置文件

程序使用`config.py`文件进行配置，主要配置项包括：

1. 数据路径配置
   ```python
   # 目标能谱数据文件路径
   TARGET_SPECTRUM = "data/target.txt"
   
   # 模板能谱数据文件路径列表
   TEMPLATE_SPECTRA = [
       "data/template1.txt",
       "data/template2.txt",
       "data/template3.txt",
   ]
   ```

2. MCMC参数
   ```python
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
   ```

3. 先验分布配置
   ```python
   # 各模板的先验分布设置，每个模板可以设置不同的先验分布
   PRIORS = [
       # 模板1的先验分布
       {
           "type": "uniform",
           "params": {"min": 0.0, "max": 10.0}
       },
       # 模板2的先验分布
       {
           "type": "normal",
           "params": {"mu": 1.0, "sigma": 0.5}
       },
       # 模板3的先验分布
       {
           "type": "lognormal",
           "params": {"mu": 0.0, "sigma": 0.5}
       },
   ]
   
   # 如果PRIORS中没有为每个模板指定先验，则使用以下默认先验
   DEFAULT_PRIOR = {
       "type": "uniform",
       "params": {"min": 0.0, "max": 10.0}
   }
   ```

4. 误差处理和HTML报告选项
   ```python
   # 误差处理方式: 'target', 'template', 'both'
   ERROR_HANDLING = "both"
   
   # 是否生成HTML报告
   HTML_REPORT = True
   
   # HTML报告中是否包含交互式图表
   INTERACTIVE_PLOTS = True
   ```

## 数据格式

输入文件应为ASCII文本文件，每行包含三个值：能量、计数值和标准差。
所有能谱(目标和模板)必须具有相同的能量轴(相同的能量点和能量范围)。

示例：
```
1.0 100.5 10.0
2.0 95.2  9.8
3.0 85.7  9.3
...
```

## 输出结果

程序将在指定的输出目录中生成以下内容：

- 参数后验分布图
- MCMC链迹图
- 拟合结果图
- 残差图
- 模板贡献饼图
- 误差贡献分析图
- HTML报告(如果启用)
- 交互式图表(如果启用)

## 示例

运行示例脚本，将生成合成数据并进行分析：

```bash
python example.py
```

示例脚本会创建一个`example_config.py`文件，供您参考如何配置不同的先验分布和其他参数。

## 自定义先验分布

为每个模板设置不同的先验分布：

```python
# config.py
PRIORS = [
    # 均匀分布先验
    {
        "type": "uniform",
        "params": {"min": 0.0, "max": 5.0}
    },
    # 正态分布先验
    {
        "type": "normal",
        "params": {"mu": 1.0, "sigma": 0.5}
    },
    # 对数正态分布先验
    {
        "type": "lognormal",
        "params": {"mu": 0.0, "sigma": 0.5}
    },
]
```

## 开发者信息

如需扩展功能，可以修改以下文件：

- `model.py` - 贝叶斯模型定义，包括先验、似然和后验函数
- `mcmc_sampler.py` - MCMC采样实现
- `analyzer.py` - 结果分析和统计计算
- `visualizer.py` - 图表生成和可视化
- `html_reporter.py` - HTML报告生成 