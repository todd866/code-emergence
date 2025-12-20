#!/usr/bin/env python3
"""Create schematic figure showing A → bottleneck → B architecture."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

# Use LaTeX-compatible fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['figure.figsize'] = (8, 3)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

fig, ax = plt.subplots()

# System A box
ax.add_patch(FancyBboxPatch((0.5, 0.3), 2, 1.4, boxstyle="round,pad=0.05",
                             facecolor='#cce5ff', edgecolor='black', linewidth=1.5))
ax.text(1.5, 1.1, 'System A', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(1.5, 0.75, r'$N = 64$ oscillators', ha='center', va='center', fontsize=9)
ax.text(1.5, 0.5, r'(autonomous)', ha='center', va='center', fontsize=9, style='italic')

# Bottleneck
ax.add_patch(FancyBboxPatch((3.5, 0.2), 1.2, 1.6, boxstyle="round,pad=0.02",
                             facecolor='#e6e6e6', edgecolor='black', linewidth=1.5))
ax.text(4.1, 1.2, 'Code', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(4.1, 0.9, r'$C_k$', ha='center', va='center', fontsize=11)
ax.text(4.1, 0.55, r'$k$ modes', ha='center', va='center', fontsize=9)

# System B box
ax.add_patch(FancyBboxPatch((5.7, 0.3), 2, 1.4, boxstyle="round,pad=0.05",
                             facecolor='#ffe6e6', edgecolor='black', linewidth=1.5))
ax.text(6.7, 1.1, 'System B', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(6.7, 0.75, r'$N = 64$ oscillators', ha='center', va='center', fontsize=9)
ax.text(6.7, 0.5, r'$N_{\mathrm{eff}}(B)$ constrained', ha='center', va='center', fontsize=9)

# Arrows
ax.annotate('', xy=(3.5, 1.0), xytext=(2.5, 1.0),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(5.7, 1.0), xytext=(4.7, 1.0),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Arrow labels
ax.text(3.0, 1.35, 'Fourier\nprojection', ha='center', va='bottom', fontsize=8)
ax.text(5.2, 1.35, r'coupling $\lambda$', ha='center', va='bottom', fontsize=8)

# Top labels
ax.text(1.5, 2.0, 'Driving', ha='center', va='center', fontsize=10, style='italic')
ax.text(4.1, 2.0, 'Bandwidth k', ha='center', va='center', fontsize=10, style='italic')
ax.text(6.7, 2.0, 'Responding', ha='center', va='center', fontsize=10, style='italic')

# Bottom annotation
ax.text(4.1, -0.2, 'Low-dimensional constraint:\nonly k modes transmitted',
        ha='center', va='top', fontsize=9, style='italic')

ax.set_xlim(0, 8.2)
ax.set_ylim(-0.6, 2.3)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('fig_schematic.pdf')
plt.savefig('fig_schematic.png')
print("Saved fig_schematic.pdf")
