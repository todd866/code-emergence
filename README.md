# The Code-Constraint Problem in Biological Systems

How Low-Dimensional Interfaces Shape High-Dimensional Dynamics

Manuscript for submission to *Progress in Biophysics & Molecular Biology*.

## Overview

This paper proposes a framework for understanding dimensional mismatch in biological systems. Low-dimensional interfaces between coupled high-dimensional systems function as **stabilizing constraints** rather than information channels.

Using coupled oscillator and gene regulatory network simulations, we demonstrate that bandwidth-limited coupling produces:
1. **Complexity collapse**: Responding system's effective dimensionality decreases with code bandwidth
2. **Bounded tracking**: Alignment remains stable despite information loss
3. **Structure dependence**: Effect requires coherent projections; random projections fail
4. **Dynamics independence**: Same signature in Kuramoto oscillators and GRN models

## Repository Structure

```
├── code_formation.tex        # Main manuscript (LaTeX)
├── code_formation.pdf        # Compiled manuscript (22 pages)
├── cover_letter.pdf          # PBMB cover letter
├── references.bib            # Bibliography
├── code/
│   ├── generate_figures.js       # Main simulation (Node.js, N=64)
│   ├── plot_figures.py           # Figure plotting (Python)
│   ├── supplementary_simulations.py  # All supplementary sims
│   ├── create_schematic.py       # Figure 1 schematic
│   └── create_reconstruction_comparison.py  # Reconstruction figure
└── figures/
    ├── fig_*.pdf             # Main manuscript figures
    └── fig_supp_*.pdf        # Supplementary figures
```

## Running the Simulations

### Requirements
- Node.js (for `generate_figures.js`)
- Python 3 with matplotlib, numpy, scipy

### Main figures (N=64, fast)
```bash
cd code
node generate_figures.js
python plot_figures.py
```

### Supplementary simulations
```bash
python supplementary_simulations.py --all    # All supplementary (~2 min)
python supplementary_simulations.py --large  # Large-scale Kuramoto (N=512-1024)
python supplementary_simulations.py --grn    # Gene regulatory network (N=256)
python supplementary_simulations.py --metrics  # Alternative complexity metrics
```

## Key Results

### Kuramoto Oscillators (N=64-1024)
| Bandwidth k | Neff(B) | Mismatch |
|-------------|---------|----------|
| 1           | 10.9    | 0.50     |
| 32          | 16.7    | 0.38     |

### Gene Regulatory Networks (N=256)
| Bandwidth k | Neff(B) | Collapse |
|-------------|---------|----------|
| 1           | 192     | 0%       |
| 32          | 121     | 37%      |

## Key Parameters

- `N`: System size (64 for main, 256-1024 for supplementary)
- `K = 0.5`: Internal coupling strength
- `λ = 1.0-5.0`: Cross-system coupling strength
- `σ = 0.03-0.3`: Noise amplitude
- `k = 1..N/2`: Code bandwidth

## Citation

If you use this code, please cite:

> Todd, I. (2025). The code-constraint problem in biological systems: How low-dimensional interfaces shape high-dimensional dynamics. *Progress in Biophysics & Molecular Biology* (in preparation).

## License

MIT License
