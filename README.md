# The Code-Constraint Problem in Biological Systems

How Low-Dimensional Interfaces Shape High-Dimensional Dynamics

Manuscript for submission to *Progress in Biophysics & Molecular Biology*.

## Overview

This paper proposes a framework for understanding dimensional mismatch in biological systems. Low-dimensional interfaces between coupled high-dimensional systems function as **stabilizing constraints** rather than information channels.

Using coupled oscillator simulations, we demonstrate that bandwidth-limited coupling produces:
1. **Complexity collapse**: Responding system's effective dimensionality decreases with code bandwidth
2. **Bounded tracking**: Alignment remains stable despite information loss
3. **Structure dependence**: Effect requires coherent projections; random projections fail

## Repository Structure

```
├── code_formation.tex        # Main manuscript (LaTeX)
├── code_formation.pdf        # Compiled manuscript
├── references.bib            # Bibliography
├── code/
│   ├── generate_figures.js   # Main simulation (Node.js)
│   ├── plot_figures.py       # Figure plotting (Python)
│   ├── create_schematic.py   # Figure 1 schematic
│   └── create_reconstruction_comparison.py
└── figures/
    └── fig_*.pdf             # Generated figures
```

## Running the Simulations

### Requirements
- Node.js (for `generate_figures.js`)
- Python 3 with matplotlib, numpy (for plotting)

### Generate data and figures
```bash
cd code
node generate_figures.js
python plot_figures.py
python create_schematic.py
```

## Key Parameters

- `N = 64`: Number of oscillators per lattice
- `K = 0.5`: Internal coupling strength
- `λ = 1.0`: Cross-system coupling strength (default)
- `σ = 0.3`: Noise amplitude
- `k = 1..32`: Code bandwidth (Fourier modes)

## Citation

If you use this code, please cite:

> Todd, I. (2025). The code-constraint problem in biological systems: How low-dimensional interfaces shape high-dimensional dynamics. *Progress in Biophysics & Molecular Biology* (in preparation).

## License

MIT License
