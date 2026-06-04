# MEBC: Memory-Enhanced Behavioural Change in Epidemic Models

Code and data for the paper:

**"Identifying memory mechanisms in Bayesian models of behavioural change during epidemics"**

Yicheng Mao<sup>1,2,*</sup>, Rob Deardon<sup>2,3</sup>, Lorna E. Deeth<sup>4</sup>

<sup>1</sup> Department of Data Analytics and Digitalization, Maastricht University  
<sup>2</sup> Department of Mathematics and Statistics, University of Calgary  
<sup>3</sup> Faculty of Veterinary Medicine, University of Calgary  
<sup>4</sup> Department of Mathematics and Statistics, University of Guelph  
<sup>*</sup> Correspondence: yicheng.mao1@ucalgary.ca

## Overview

This repository implements the Memory Mechanism Enhanced Behavioural Change (MEBC) framework, which extends the discrete-time stochastic SIR model by incorporating population-level behavioural responses to epidemic severity. The framework introduces an alarm function driven by past infection counts, where different memory mechanisms control how historical information is weighted:

| Mechanism | Description |
|-----------|-------------|
| **Memoryless** | Only the most recent time step informs behaviour |
| **Sliding window** | A fixed window of *k*<sub>max</sub> past time steps, equally weighted |
| **Power-law** | All past time steps, with weights decaying as *j*<sup>−λ<sub>P</sub></sup> |
| **Exponential** | All past time steps, with weights decaying as exp(−λ<sub>E</sub> *j*) |
| **Reciprocal** | All past time steps, with weights decaying as 1/(1 + λ<sub>R</sub> *j*) |

Inference is performed via data-augmented MCMC with adaptive Metropolis-Hastings proposals (online covariance estimation). Model comparison uses WAIC, and convergence is assessed with the Gelman–Rubin–Brooks diagnostic.

## Repository Structure

```
MEBC/
├── Simulation/                        # Simulation study
│   ├── simulation data generation/    # Generate synthetic epidemics & figures
│   ├── memoryless/                    # Fit memoryless model to simulated data
│   ├── sliding/                       # Fit sliding window model (two-stage: select k_max, then full MCMC)
│   │   ├── stage 1/                   #   Stage 1: k_max selection
│   │   └── stage2/                    #   Stage 2: full inference with fixed k_max
│   ├── power/                         # Fit power-law model
│   ├── exp/                           # Fit exponential model
│   └── rep/                           # Fit reciprocal model
│
├── applicationManitoba/               # Real-data application: influenza in Manitoba, Canada
│   ├── noBC/                          # Baseline SIR (no behavioural change)
│   ├── memoryless/
│   ├── sliding/
│   ├── power/
│   ├── exp/
│   └── rep/
│
└── applicationMiami/                  # Real-data application: COVID-19 in Miami-Dade County
    ├── Miami_daily_cases_ma7.csv      # Observed case data (7-day moving average)
    ├── memoryless/
    ├── sliding/
    ├── power/
    ├── exp/
    ├── rep/
    └── plot/                          # Posterior predictive plots and R_t estimates
```

Within each model folder, the typical file structure is:

- `functions.jl` — model-specific functions (likelihood, alarm computation, MCMC sampler, diagnostics)
- `chain1.ipynb`, `chain2.ipynb`, `chain3.ipynb` — three independent MCMC chains (1M iterations each)
- `postmcmc.ipynb` — posterior summaries, convergence diagnostics, and WAIC computation
- `data/` — intermediate CSV outputs (posterior summaries, WAIC values, predicted incidence)

## Requirements

- [Julia](https://julialang.org/) (≥ 1.9)
- Julia packages: `Distributions`, `StatsBase`, `Statistics`, `LinearAlgebra`, `Printf`, `Dates`, `CSV`, `DataFrames`
- [Jupyter](https://jupyter.org/) with the [IJulia](https://github.com/JuliaLang/IJulia.jl) kernel

## Usage

1. **Simulation study**: Start with `Simulation/simulation data generation/` to generate synthetic epidemic data, then run the chain notebooks in each mechanism subfolder.

2. **Real-data applications**: Run `chain1.ipynb` through `chain3.ipynb` in each model subfolder (each chain takes approximately 1 hour on a 16-core CPU), then `postmcmc.ipynb` for posterior analysis.

3. **Sliding window model**: This model requires a two-stage procedure. Run stage 1 (`stage 1/`) to identify the optimal *k*<sub>max</sub> via profile analysis, then run stage 2 with the selected *k*<sub>max</sub> fixed.

