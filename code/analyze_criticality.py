#!/usr/bin/env python3
"""
Post-process critical scaling data to extract:
1. χ peaks and locations
2. χ_max vs N scaling
3. Order parameter definition

Run after critical_scaling_large.py finishes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11

# Load data
try:
    data = np.genfromtxt('../figures/critical_scaling.csv', delimiter=',', names=True)
    print("Loaded critical_scaling.csv")
except:
    data = np.genfromtxt('../figures/finite_size_scaling.csv', delimiter=',', names=True)
    print("Loaded finite_size_scaling.csv (fallback)")

N_values = sorted(list(set(data['N'])))
print(f"N values: {N_values}")

# =============================================================================
# Recompute order parameter and susceptibility properly
# =============================================================================

# For each N, find:
# - φ(k/N) = 1 - Neff(B) / baseline
# - χ(k/N) = susceptibility (from the data, or variance proxy)

results_by_N = {}
for N in N_values:
    mask = data['N'] == N
    k_over_N = data['k_over_N'][mask]
    neff_B = data['neff_B'][mask]
    baseline = data['baseline'][mask][0]

    # Order parameter
    phi = 1 - neff_B / baseline

    # Susceptibility - use the 'susceptibility' column if available
    if 'susceptibility' in data.dtype.names:
        chi = data['susceptibility'][mask]
    else:
        # Fallback: use neff_B_std as proxy
        chi = data['neff_B_std'][mask] ** 2

    results_by_N[N] = {
        'k_over_N': k_over_N,
        'phi': phi,
        'chi': chi,
        'neff_B': neff_B,
        'baseline': baseline
    }

# =============================================================================
# Find χ peaks
# =============================================================================

chi_max = []
chi_max_location = []

for N in N_values:
    r = results_by_N[N]
    idx_max = np.argmax(r['chi'])
    chi_max.append(r['chi'][idx_max])
    chi_max_location.append(r['k_over_N'][idx_max])

chi_max = np.array(chi_max)
chi_max_location = np.array(chi_max_location)
N_arr = np.array(N_values)

print("\nχ peak analysis:")
for i, N in enumerate(N_values):
    print(f"  N={int(N):5d}: χ_max = {chi_max[i]:8.2f} at k/N = {chi_max_location[i]:.4f}")

# =============================================================================
# Fit χ_max ~ N^α
# =============================================================================

# Log-log fit
log_N = np.log(N_arr)
log_chi = np.log(chi_max)

# Only fit if we have enough points and chi_max is increasing
if len(N_values) >= 3 and chi_max[-1] > chi_max[0]:
    slope, intercept = np.polyfit(log_N, log_chi, 1)
    alpha = slope

    # Compute R²
    predicted = intercept + slope * log_N
    ss_res = np.sum((log_chi - predicted)**2)
    ss_tot = np.sum((log_chi - np.mean(log_chi))**2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\nScaling exponent fit:")
    print(f"  χ_max ~ N^{alpha:.3f}")
    print(f"  R² = {r_squared:.4f}")

    if alpha > 0:
        print(f"  → Susceptibility GROWS with N (consistent with critical-like transition)")
    else:
        print(f"  → Susceptibility decreases with N (no criticality)")
else:
    alpha = None
    print("\nInsufficient data or non-monotonic χ_max - skipping exponent fit")

# =============================================================================
# Create publication figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_values)))

# Panel A: Order parameter φ vs k/N
ax = axes[0, 0]
for i, N in enumerate(N_values):
    r = results_by_N[N]
    ax.plot(r['k_over_N'], r['phi'], 'o-', color=colors[i],
            label=f'N={int(N)}', markersize=5, linewidth=1.5)
ax.set_xlabel(r'$k/N$')
ax.set_ylabel(r'Order parameter $\phi = 1 - N_{\mathrm{eff}}/N_{\mathrm{eff}}^0$')
ax.set_xscale('log')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('(A) Order parameter')
ax.grid(True, alpha=0.3)

# Panel B: Susceptibility χ vs k/N
ax = axes[0, 1]
for i, N in enumerate(N_values):
    r = results_by_N[N]
    ax.plot(r['k_over_N'], r['chi'], 'o-', color=colors[i],
            label=f'N={int(N)}', markersize=5, linewidth=1.5)
ax.set_xlabel(r'$k/N$')
ax.set_ylabel(r'Susceptibility $\chi$')
ax.set_xscale('log')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('(B) Susceptibility peaks')
ax.grid(True, alpha=0.3)

# Mark peaks
for i, N in enumerate(N_values):
    r = results_by_N[N]
    idx_max = np.argmax(r['chi'])
    ax.scatter(r['k_over_N'][idx_max], r['chi'][idx_max],
               color=colors[i], s=100, marker='*', edgecolor='black', zorder=10)

# Panel C: χ_max vs N (the key plot)
ax = axes[1, 0]
ax.scatter(N_arr, chi_max, c=colors, s=100, zorder=5)

if alpha is not None:
    # Plot fit line
    N_fit = np.linspace(N_arr.min() * 0.8, N_arr.max() * 1.2, 100)
    chi_fit = np.exp(intercept) * N_fit ** alpha
    ax.plot(N_fit, chi_fit, 'k--', linewidth=2,
            label=rf'$\chi_{{\max}} \sim N^{{{alpha:.2f}}}$ (R²={r_squared:.2f})')
    ax.legend(loc='lower right')

ax.set_xlabel(r'System size $N$')
ax.set_ylabel(r'Peak susceptibility $\chi_{\max}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('(C) Susceptibility scaling')
ax.grid(True, alpha=0.3)

# Panel D: Peak location vs N
ax = axes[1, 1]
ax.scatter(N_arr, chi_max_location, c=colors, s=100, zorder=5)
ax.set_xlabel(r'System size $N$')
ax.set_ylabel(r'Peak location $(k/N)_c$')
ax.set_xscale('log')
ax.set_title('(D) Critical point drift')
ax.grid(True, alpha=0.3)

# Add horizontal line at mean if locations are stable
if np.std(chi_max_location) / np.mean(chi_max_location) < 0.3:
    ax.axhline(np.mean(chi_max_location), color='gray', linestyle='--',
               label=f'mean = {np.mean(chi_max_location):.4f}')
    ax.legend()

plt.tight_layout()
plt.savefig('../figures/fig_susceptibility_scaling.pdf', dpi=300)
plt.savefig('../figures/fig_susceptibility_scaling.png', dpi=150)
print("\nSaved fig_susceptibility_scaling.pdf/png")

# =============================================================================
# Summary for paper
# =============================================================================

print("\n" + "="*70)
print("SUMMARY FOR PAPER")
print("="*70)

if alpha is not None and alpha > 0 and r_squared > 0.8:
    print("""
RESULT: Clean susceptibility scaling detected.

Suggested text for §5.5 or Supplement:

  "We define an order parameter φ = 1 − N_eff(B)/N_eff^baseline and
  susceptibility χ as the variance of φ across independent trials.
  The susceptibility exhibits a pronounced peak (Figure SX, panel B) whose
  height grows with system size as χ_max ~ N^{%.2f} (R² = %.2f, panel C).
  This scaling is consistent with a sharpening continuous transition,
  though we do not claim specific universality-class exponents."
""" % (alpha, r_squared))
elif alpha is not None:
    print(f"""
RESULT: Susceptibility scaling detected but weak (α = {alpha:.2f}, R² = {r_squared:.2f}).

Recommendation: Include as supplement but don't emphasize exponents.
""")
else:
    print("""
RESULT: No clean susceptibility scaling.

Recommendation: Keep current "finite-size sharpening" claim without exponents.
""")
