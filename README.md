# Low-Dimensional Codes Constrain High-Dimensional Biological Dynamics

Manuscript and supporting materials for submission to *Progress in Biophysics & Molecular Biology*.

## Overview

This repository contains code and data for simulations demonstrating that low-dimensional codes act as dimensional constraints between coupled high-dimensional dynamical systems. Using Kuramoto oscillator lattices with bandwidth-limited Fourier coupling, we show that:

1. Reducing code bandwidth induces systematic complexity collapse in the responding system
2. The driving system's complexity remains unchanged regardless of bandwidth
3. Random k-mode projections of the same dimensionality fail to constrain complexity
4. Constraining codes must capture coherent macroscopic degrees of freedom

## Repository Structure

```
├── code_formation_jtb.tex    # Main manuscript (LaTeX)
├── code_formation_jtb.pdf    # Compiled manuscript
├── references.bib            # Bibliography
├── highlights.txt            # Highlights (3-5 bullets)
└── supporting_files/
    ├── generate_figures.js   # Main simulation code (Node.js)
    ├── plot_figures.py       # Figure generation (Python/matplotlib)
    ├── create_schematic.py   # Figure 1 schematic
    ├── create_reconstruction_comparison.py  # Figure 5 reconstruction demo
    ├── fig3_complexity_vs_k.csv    # Data: complexity vs bandwidth
    ├── fig4_complexity_vs_lambda.csv  # Data: complexity vs coupling strength
    ├── fig_random_projection.csv   # Data: random mode control
    └── fig_*.pdf             # Generated figures
```

## Running the Simulations

### Requirements
- Node.js (for `generate_figures.js`)
- Python 3 with matplotlib, numpy (for plotting)

### Generate data and figures
```bash
cd supporting_files
node generate_figures.js
python plot_figures.py
python create_schematic.py
python create_reconstruction_comparison.py
```

## Key Parameters

- `N = 64`: Number of oscillators per lattice
- `K = 0.5`: Internal coupling strength
- `λ = 1.0`: Cross-system coupling strength (default)
- `σ = 0.3`: Noise amplitude
- `k = 1..32`: Code bandwidth (Fourier modes)

## Citation

If you use this code, please cite:

> Todd, I. (2025). Low-dimensional codes constrain high-dimensional biological dynamics. *Progress in Biophysics & Molecular Biology* (in preparation).

## License

MIT License
