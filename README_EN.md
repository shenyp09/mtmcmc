# Multi Template MCMC Bayesian Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Multi Template MCMC Bayesian Analysis is a Python package for spectral fitting using Markov Chain Monte Carlo (MCMC) methods for Bayesian analysis, capable of simultaneously fitting multiple template spectra.

[中文文档](README.md)

## Features

- Support for simultaneous fitting of multiple spectral templates
- Efficient MCMC sampling based on emcee
- Multiple prior distribution options (uniform, normal, lognormal, truncated normal)
- Comprehensive posterior distribution and fitting result analysis
- Generation of visually appealing charts
- Bilingual HTML report generation (Chinese and English)
- Interactive chart support (using Plotly)
- Automatic timestamp output directories
- Detailed parameter statistics analysis (mean, median, variance, various confidence intervals)
- Scientific notation display for fitting parameters

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/mtmcmc.git
cd mtmcmc
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Edit the `config.py` file to set data paths, MCMC parameters, and other options
2. Run the main program:

```bash
python mtmcmc.py
```

### Using the Example Script

The project includes an example script that can generate synthetic data and run the analysis:

```bash
python example.py
```

## Configuration Parameters

Edit the `config.py` file to configure the following parameters:

### Data Path Configuration
- `TARGET_SPECTRUM`: Path to the target spectrum data file
- `TEMPLATE_SPECTRA`: List of paths to template spectrum data files

### Output Configuration
- `OUTPUT_DIR`: Results output directory
- `ADD_TIMESTAMP`: Whether to add a timestamp subdirectory in the output directory

### MCMC Parameter Configuration
- `NWALKERS`: Number of MCMC walkers
- `NSTEPS`: Number of MCMC sampling steps
- `BURNIN`: Number of MCMC burn-in steps
- `PROGRESS`: Whether to display a progress bar
- `NCORES`: Number of CPU cores to use, None means use all available cores

### Prior Distribution Configuration
- `PRIORS`: Prior distribution settings for each template, supporting the following distribution types:
  - `uniform`: Uniform distribution, parameters are min and max
  - `normal`: Normal distribution, parameters are mu and sigma
  - `lognormal`: Lognormal distribution, parameters are mu and sigma
  - `truncnorm`: Truncated normal distribution, parameters are min, max, mu, and sigma
- `DEFAULT_PRIOR`: Default prior distribution settings

### Error Handling and HTML Report Options
- `ERROR_HANDLING`: Error handling method ('target', 'template', or 'both')
- `HTML_REPORT`: Whether to generate an HTML report
- `INTERACTIVE_PLOTS`: Whether to include interactive charts in the HTML report
- `TEMPLATE_DIR`: HTML template directory
- `HTML_LANGUAGES`: HTML report language settings, options: ["zh"], ["en"], ["zh", "en"]

## Module Structure

- `mtmcmc.py`: Main program
- `data_loader.py`: Data loading and preprocessing module
- `model.py`: Model definition and prior distribution module
- `mcmc_sampler.py`: MCMC sampling module
- `analyzer.py`: Result analysis module
- `visualizer.py`: Visualization module (supports bilingual charts)
- `html_reporter.py`: HTML report generation module (supports bilingual reports)
- `config.py`: Configuration file
- `example.py`: Example script

## Data Format

Input data files should be in text format, with three columns per line: energy, count, error. For example:

```
0.0 10.5 1.2
0.1 11.2 1.3
...
```

## Result Analysis

Analysis results will be saved in the output directory, including:

- Posterior distribution of template weights
- Fitting results and residual analysis
- Template contribution analysis
- Error contribution analysis
- Parameter statistics analysis (mean, median, variance, various confidence intervals)
- Comprehensive HTML report (Chinese and/or English)

### Parameter Statistics Analysis

The following statistics are provided for each fitting parameter:
- Median value
- Mean value
- Standard deviation
- Variance
- MAP estimate (Maximum A Posteriori)
- 68% confidence interval
- 95% confidence interval
- 99.7% confidence interval
- Upper and lower error ranges

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contribution

Contributions of code, issue reports, or improvement suggestions are welcome. 