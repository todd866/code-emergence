# Quotient Geometry of Statistical Manifolds Under Dimensional Collapse

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**When a statistical manifold is observed through a lower-dimensional map, what geometric structure survives?**

Manuscript for **Information Geometry** (Springer).

## Overview

This paper develops the quotient geometry of statistical manifolds under collapse maps π: M → ℝᵏ with k < dim(M). We characterize when the Fisher metric descends to the quotient, how α-connections transform, and what determines the number of distinguishable equivalence classes at finite resolution.

### Main Results

1. **Fiber Structure Theorem**: Collapse maps foliate M into fibers along which the observed Fisher metric degenerates. Points on the same fiber yield identical observed distributions and are statistically non-identifiable (connecting to Watanabe's singular learning theory).

2. **Quotient Metric Theorem**: The Fisher metric descends to a Riemannian metric on M/∼ **if and only if** the metric is *bundle-like* (constant along fibers on horizontal vectors). Totally geodesic fibers provide a sufficient condition. The α-connection structure descends under additional conditions.

3. **Covering Number Bounds**: The number of ε-distinguishable classes scales as:
   ```
   N(ε) ≥ C_K · ε^{-r}
   ```
   where r is the projection rank. This quantifies the "effective alphabet size" induced by collapse.

## Companion Paper

This paper forms a two-paper program with:

**[Minimal Embedding Dimension for Recurrent Processes](https://github.com/todd866/minimalembeddingdimension)** (INGE-D-25-00099, under review)

- **Companion paper**: establishes a specific phenomenon—cyclic processes with monotone meta-time require k ≥ 3 for self-intersection-free embedding
- **This paper**: provides the general quotient-geometric framework that makes that phenomenon inevitable; the minimal embedding result emerges as Corollary 13

The papers share notation: φ for phase coordinate, τ for meta-time, V = ker(dπ) for vertical distribution, H = V^⊥ for horizontal.

## Key Connections

- **Chentsov's theorem**: The Fisher metric is the unique metric invariant under sufficient statistics; our quotient construction respects this.
- **Watanabe's singular learning theory**: Fiber structure provides the differential-geometric setting for his algebraic singularities.
- **Amari's α-geometry**: We characterize when dual connections survive collapse.

## Repository Structure

```
├── code_emergence.tex        # Main manuscript
├── code_emergence.pdf        # Compiled PDF
├── cover_letter.tex          # Submission cover letter
├── cover_letter.pdf          # Compiled cover letter
└── references.bib            # Bibliography
```

## Status

| Item | Status |
|------|--------|
| Target | Information Geometry |
| Companion | [minimalembeddingdimension](https://github.com/todd866/minimalembeddingdimension) (under review) |
| Strategy | Submit after companion decision |

## Citation

```bibtex
@article{todd2025quotient,
  title={Quotient Geometry of Statistical Manifolds Under Dimensional Collapse},
  author={Todd, Ian},
  journal={Information Geometry},
  year={2025},
  note={In preparation}
}
```

## License

MIT License
