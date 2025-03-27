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

In `config.py`, you can configure the following MCMC-related parameters:

```python
# Number of MCMC walkers
NWALKERS = 32

# Number of MCMC steps
NSTEPS = 5000

# Number of burn-in steps
BURNIN = 1000

# Whether to show progress bar
PROGRESS = True

# Number of CPU cores to use, None means use all available cores
NCORES = None

# MCMC move strategy configuration
# Each element is a tuple (move, weight), where weight represents the probability of using this move
# Available move strategies:
# - emcee.moves.DESnookerMove(): Differential Evolution Snooker move
# - emcee.moves.DEMove(): Differential Evolution move
# - emcee.moves.GaussianMove(): Gaussian move
# - emcee.moves.KDEMove(): Kernel Density Estimation move
# - emcee.moves.StretchMove(): Stretch move
MCMC_MOVES = [
    (emcee.moves.DESnookerMove(), 0.8),  # Use 80% Snooker move
    (emcee.moves.DEMove(), 0.2),          # Use 20% Differential Evolution move
]
```

MCMC move strategies (moves) are crucial settings that control how the sampler explores the parameter space. The program provides several move strategies to choose from:

1. **DESnookerMove**: Differential Evolution Snooker move
   - Advantages: Excellent exploration capability in high-dimensional parameter spaces
   - Suitable for: Complex high-dimensional parameter spaces

2. **DEMove**: Differential Evolution move
   - Advantages: Combines information from multiple walkers
   - Suitable for: Sampling requiring walker collaboration

3. **GaussianMove**: Gaussian move
   - Advantages: Simple and efficient
   - Suitable for: Simple parameter spaces

4. **KDEMove**: Kernel Density Estimation move
   - Advantages: Can adapt to complex posterior distributions
   - Suitable for: Multi-modal or non-Gaussian distributions

5. **StretchMove**: Stretch move
   - Advantages: Low computational overhead
   - Suitable for: Cases requiring fast sampling

You can configure different combinations of move strategies by adjusting the `MCMC_MOVES` list. Each move strategy has a weight value that represents the probability of using that strategy. The weights should satisfy:

```python
sum(weight for _, weight in MCMC_MOVES) == 1.0
```

The default configuration uses 80% Snooker move and 20% Differential Evolution move, which provides good sampling performance in most cases. You can adjust these weights or add other move strategies based on your specific problem.

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