#!/usr/bin/env python3
"""
Generate publication-quality figures for JTB paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use LaTeX-compatible fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.figsize'] = (5, 4)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# Load data from figures directory
fig3_data = np.genfromtxt('../figures/fig3_complexity_vs_k.csv', delimiter=',', names=True)
fig4_data = np.genfromtxt('../figures/fig4_complexity_vs_lambda.csv', delimiter=',', names=True)

# =============================================================================
# FIGURE 3: Complexity vs k
# =============================================================================

fig, ax = plt.subplots()

k = fig3_data['k']
Neff_A = fig3_data['Neff_A']
Neff_B = fig3_data['Neff_B']
se_A = fig3_data['se_A']
se_B = fig3_data['se_B']

ax.errorbar(k, Neff_A, yerr=se_A, marker='o', linestyle='-', color='#666666',
            label=r'System $A$ (driving)', capsize=3, markersize=6)
ax.errorbar(k, Neff_B, yerr=se_B, marker='s', linestyle='-', color='#2166ac',
            label=r'System $B$ (responding)', capsize=3, markersize=6)

ax.set_xlabel(r'Code bandwidth $k$')
ax.set_ylabel(r'Effective dimensionality $N_{\mathrm{eff}}$')
ax.set_xscale('log', base=2)
ax.set_xticks(k)
ax.set_xticklabels([str(int(x)) for x in k])
ax.set_ylim(8, 20)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/fig_complexity.pdf')
plt.savefig('../figures/fig_complexity.png')
print("Saved fig_complexity.pdf")

# =============================================================================
# FIGURE 4a: Mismatch vs k
# =============================================================================

fig, ax = plt.subplots()

mismatch = fig3_data['mismatch']

ax.plot(k, mismatch, marker='o', linestyle='-', color='#b2182b', markersize=6)

# Add control line (uncoupled)
ax.axhline(y=0.64, color='#666666', linestyle='--', label='Uncoupled control')

ax.set_xlabel(r'Code bandwidth $k$')
ax.set_ylabel(r'Mismatch $\Delta$')
ax.set_xscale('log', base=2)
ax.set_xticks(k)
ax.set_xticklabels([str(int(x)) for x in k])
ax.set_ylim(0.3, 0.7)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/fig_mismatch.pdf')
plt.savefig('../figures/fig_mismatch.png')
print("Saved fig_mismatch.pdf")

# =============================================================================
# FIGURE 4b: Complexity vs lambda
# =============================================================================

fig, ax = plt.subplots()

lam = fig4_data['lambda']
Neff_B_lam = fig4_data['Neff_B']
mismatch_lam = fig4_data['mismatch']

ax2 = ax.twinx()

l1, = ax.plot(lam, Neff_B_lam, marker='s', linestyle='-', color='#2166ac',
              markersize=6, label=r'$N_{\mathrm{eff}}(B)$')
l2, = ax2.plot(lam, mismatch_lam, marker='^', linestyle='--', color='#b2182b',
               markersize=6, label='Mismatch')

ax.set_xlabel(r'Coupling strength $\lambda$')
ax.set_ylabel(r'Effective dimensionality $N_{\mathrm{eff}}(B)$', color='#2166ac')
ax2.set_ylabel(r'Mismatch $\Delta$', color='#b2182b')

ax.tick_params(axis='y', labelcolor='#2166ac')
ax2.tick_params(axis='y', labelcolor='#b2182b')

ax.set_ylim(10, 18)
ax2.set_ylim(0.3, 0.7)

lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/fig_lambda.pdf')
plt.savefig('../figures/fig_lambda.png')
print("Saved fig_lambda.pdf")

# =============================================================================
# COMBINED FIGURE (2-panel for main result)
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Panel A: Complexity
ax1.errorbar(k, Neff_A, yerr=se_A, marker='o', linestyle='-', color='#666666',
            label=r'System $A$ (driving)', capsize=3, markersize=6)
ax1.errorbar(k, Neff_B, yerr=se_B, marker='s', linestyle='-', color='#2166ac',
            label=r'System $B$ (responding)', capsize=3, markersize=6)

ax1.set_xlabel(r'Code bandwidth $k$')
ax1.set_ylabel(r'Effective dimensionality $N_{\mathrm{eff}}$')
ax1.set_xscale('log', base=2)
ax1.set_xticks(k)
ax1.set_xticklabels([str(int(x)) for x in k])
ax1.set_ylim(8, 20)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_title('(A) Complexity collapse', fontsize=11)

# Panel B: Mismatch
ax2.plot(k, mismatch, marker='o', linestyle='-', color='#b2182b', markersize=6)
ax2.axhline(y=0.64, color='#666666', linestyle='--', label='Uncoupled control')

ax2.set_xlabel(r'Code bandwidth $k$')
ax2.set_ylabel(r'Mismatch $\Delta$')
ax2.set_xscale('log', base=2)
ax2.set_xticks(k)
ax2.set_xticklabels([str(int(x)) for x in k])
ax2.set_ylim(0.3, 0.7)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_title('(B) Tracking error', fontsize=11)

plt.tight_layout()
plt.savefig('../figures/fig_main_result.pdf')
plt.savefig('../figures/fig_main_result.png')
print("Saved fig_main_result.pdf")

# =============================================================================
# FIGURE 5: RANDOM PROJECTION CONTROL
# =============================================================================

random_data = np.genfromtxt('../figures/fig_random_projection.csv', delimiter=',', names=True)

fig, ax = plt.subplots()

k_rand = random_data['k']
Neff_B_fourier = fig3_data['Neff_B']
Neff_B_random = random_data['Neff_B']
se_B_fourier = fig3_data['se_B']
se_B_random = random_data['se_B']

ax.errorbar(k, Neff_B_fourier, yerr=se_B_fourier, marker='s', linestyle='-', color='#2166ac',
            label=r'Low-frequency Fourier', capsize=3, markersize=6)
ax.errorbar(k_rand, Neff_B_random, yerr=se_B_random, marker='^', linestyle='--', color='#b2182b',
            label=r'Random mode selection', capsize=3, markersize=6)

ax.axhline(y=17.0, color='#666666', linestyle=':', alpha=0.7, label=r'$N_{\mathrm{eff}}(A)$ baseline')

ax.set_xlabel(r'Code bandwidth $k$')
ax.set_ylabel(r'Effective dimensionality $N_{\mathrm{eff}}(B)$')
ax.set_xscale('log', base=2)
ax.set_xticks(k)
ax.set_xticklabels([str(int(x)) for x in k])
ax.set_ylim(8, 20)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/fig_random_control.pdf')
plt.savefig('../figures/fig_random_control.png')
print("Saved fig_random_control.pdf")

print("\nAll figures generated!")
