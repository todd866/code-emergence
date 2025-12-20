#!/usr/bin/env python3
"""
Supplementary simulations for Code-Constraint paper.
Consolidates: scalable_simulation.py, grn_tuned.py, alternative_metrics.py

Run with: python3 supplementary_simulations.py [--all|--large|--grn|--metrics]

Generates:
- fig_supp_large_scale.pdf (N=512 Kuramoto)
- fig_supp_grn_tuned.pdf (Gene regulatory network)
- fig_supp_alternative_metrics.pdf (Non-Fourier metrics)
- fig_supp_phase_transition.pdf (Order parameter scan)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import time
import os

# Plotting setup
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = "../figures"

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def spectral_complexity(theta):
    """Spectral entropy of Fourier amplitudes."""
    N = len(theta)
    z = np.exp(1j * theta)
    modes = np.fft.fft(z)
    amps = np.abs(modes[1:N//2])
    total = np.sum(amps)
    if total < 1e-10:
        return 1.0
    p = amps / total
    p_nonzero = p[p > 1e-10]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    return np.exp(entropy)


def fourier_encode_decode(theta, k):
    """
    Encode/decode through first k Fourier modes.
    Matches paper Eq. 4: θ̂_i = arg(Σ_{m=0}^{k} C_m e^{i 2π m i / N})
    """
    N = len(theta)
    z = np.exp(1j * theta)
    modes = np.fft.fft(z)[:k+1]
    # Reconstruct using one-sided sum (no negative frequency mirroring)
    modes_filtered = np.zeros(N, dtype=complex)
    modes_filtered[:k+1] = modes
    z_recon = np.fft.ifft(modes_filtered)
    return np.angle(z_recon)


def kuramoto_coupled_step(theta_A, theta_B, omega_A, omega_B, k, K, lambd, noise, dt, rng):
    """Single coupled Kuramoto step with Euler-Maruyama."""
    N = len(theta_A)
    sqrt_dt = np.sqrt(dt)

    # Step A
    left_A, right_A = np.roll(theta_A, 1), np.roll(theta_A, -1)
    coupling_A = np.sin(left_A - theta_A) + np.sin(right_A - theta_A)
    theta_A = theta_A + dt * (omega_A + K * coupling_A) + sqrt_dt * noise * rng.standard_normal(N)
    theta_A = np.mod(theta_A + np.pi, 2*np.pi) - np.pi

    # Code
    target = fourier_encode_decode(theta_A, k)

    # Step B
    left_B, right_B = np.roll(theta_B, 1), np.roll(theta_B, -1)
    coupling_B = np.sin(left_B - theta_B) + np.sin(right_B - theta_B)
    code_constraint = lambd * np.sin(target - theta_B)
    theta_B = theta_B + dt * (omega_B + K * coupling_B + code_constraint) + sqrt_dt * noise * 0.5 * rng.standard_normal(N)
    theta_B = np.mod(theta_B + np.pi, 2*np.pi) - np.pi

    return theta_A, theta_B


# =============================================================================
# 1. LARGE-SCALE KURAMOTO (N=512)
# =============================================================================

def run_large_scale_kuramoto():
    """Large-scale Kuramoto simulations at N=512."""
    print("\n" + "="*70)
    print("LARGE-SCALE KURAMOTO (N=512)")
    print("="*70)

    N = 512
    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    n_trials = 8
    K, lambd, noise, dt = 0.5, 1.0, 0.3, 0.1
    burn_in, measure_steps = 500, 500
    rng = np.random.default_rng(42)

    results = {'k': [], 'Neff_B': [], 'Neff_B_std': [], 'mismatch': []}
    results_random = {'k': [], 'Neff_B': [], 'Neff_B_std': [], 'mismatch': []}

    for random_proj in [False, True]:
        target_results = results_random if random_proj else results
        label = "Random" if random_proj else "Structured"
        print(f"\n--- {label} projection ---")

        for k in k_values:
            trial_Neff_B = []
            trial_mismatch = []

            for trial in range(n_trials):
                theta_A = rng.uniform(-np.pi, np.pi, N)
                theta_B = rng.uniform(-np.pi, np.pi, N)
                omega_A = 0.5 + 0.15 * rng.standard_normal(N)
                omega_B = 0.5 + 0.15 * rng.standard_normal(N)

                if random_proj:
                    all_modes = np.arange(1, N//2 + 1)
                    rng.shuffle(all_modes)
                    random_modes = all_modes[:k]

                # Burn-in and measurement (simplified for random)
                for _ in range(burn_in + measure_steps):
                    if random_proj:
                        # Random projection version
                        sqrt_dt = np.sqrt(dt)
                        left_A, right_A = np.roll(theta_A, 1), np.roll(theta_A, -1)
                        coupling_A = np.sin(left_A - theta_A) + np.sin(right_A - theta_A)
                        theta_A = theta_A + dt * (omega_A + K * coupling_A) + sqrt_dt * noise * rng.standard_normal(N)
                        theta_A = np.mod(theta_A + np.pi, 2*np.pi) - np.pi

                        # Random mode projection
                        z = np.exp(1j * theta_A)
                        all_mds = np.fft.fft(z)
                        modes_filtered = np.zeros(N, dtype=complex)
                        for m in random_modes:
                            if m < N:
                                modes_filtered[m] = all_mds[m]
                                if N - m < N:
                                    modes_filtered[N - m] = np.conj(all_mds[m])
                        target = np.angle(np.fft.ifft(modes_filtered))

                        left_B, right_B = np.roll(theta_B, 1), np.roll(theta_B, -1)
                        coupling_B = np.sin(left_B - theta_B) + np.sin(right_B - theta_B)
                        code_constraint = lambd * np.sin(target - theta_B)
                        theta_B = theta_B + dt * (omega_B + K * coupling_B + code_constraint) + sqrt_dt * noise * 0.5 * rng.standard_normal(N)
                        theta_B = np.mod(theta_B + np.pi, 2*np.pi) - np.pi
                    else:
                        theta_A, theta_B = kuramoto_coupled_step(
                            theta_A, theta_B, omega_A, omega_B, k, K, lambd, noise, dt, rng)

                trial_Neff_B.append(spectral_complexity(theta_B))
                trial_mismatch.append(np.mean(np.abs(np.sin((theta_A - theta_B)/2))))

            target_results['k'].append(k)
            target_results['Neff_B'].append(np.mean(trial_Neff_B))
            target_results['Neff_B_std'].append(np.std(trial_Neff_B) / np.sqrt(n_trials))
            target_results['mismatch'].append(np.mean(trial_mismatch))

            print(f"  k={k:3d}: Neff(B)={target_results['Neff_B'][-1]:.1f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(results['k'], results['Neff_B'], yerr=results['Neff_B_std'],
                 marker='s', linestyle='-', color='#2166ac', label='Structured (Fourier)', capsize=3)
    ax1.errorbar(results_random['k'], results_random['Neff_B'], yerr=results_random['Neff_B_std'],
                 marker='^', linestyle='--', color='#b2182b', label='Random projection', capsize=3)
    ax1.set_xlabel('Code bandwidth $k$')
    ax1.set_ylabel(r'Effective dimensionality $N_{\mathrm{eff}}(B)$')
    ax1.set_xscale('log', base=2)
    ax1.legend()
    ax1.set_title(f'(A) Kuramoto oscillators, N={N}')
    ax1.grid(True, alpha=0.3)

    ax2.plot(results['k'], results['mismatch'], 's-', color='#2166ac', label='Structured')
    ax2.plot(results_random['k'], results_random['mismatch'], '^--', color='#b2182b', label='Random')
    ax2.set_xlabel('Code bandwidth $k$')
    ax2.set_ylabel('Mismatch $\\Delta$')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.set_title('(B) Tracking error')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_supp_large_scale.pdf')
    print(f"\nSaved fig_supp_large_scale.pdf")


# =============================================================================
# 2. GENE REGULATORY NETWORK
# =============================================================================

def run_grn():
    """Gene regulatory network simulations."""
    print("\n" + "="*70)
    print("GENE REGULATORY NETWORK (N=256)")
    print("="*70)

    from scipy.fft import dct, idct

    N = 256
    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    n_trials = 10
    tau, lambd, noise, dt = 3.0, 5.0, 0.03, 0.02
    burn_in, measure_steps = 2000, 1000
    sparsity = 0.15
    rng = np.random.default_rng(42)

    def sigmoid(x, steepness=5.0):
        return 1.0 / (1.0 + np.exp(-steepness * x))

    def grn_complexity(x):
        X = dct(x, type=2, norm='ortho')
        amps = np.abs(X[1:])
        total = np.sum(amps)
        if total < 1e-10:
            return 1.0
        p = amps / total
        p_nonzero = p[p > 1e-10]
        return np.exp(-np.sum(p_nonzero * np.log(p_nonzero)))

    def pca_encode_decode(x, k):
        X = dct(x, type=2, norm='ortho')
        X_filtered = np.zeros_like(X)
        X_filtered[:k] = X[:k]
        return idct(X_filtered, type=2, norm='ortho')

    results = {'k': [], 'Neff_A': [], 'Neff_B': [], 'Neff_A_std': [], 'Neff_B_std': [], 'mismatch': []}

    for k in k_values:
        trial_A, trial_B, trial_mm = [], [], []

        for trial in range(n_trials):
            W_base = rng.standard_normal((N, N)) * (rng.random((N, N)) < sparsity)
            W_base = W_base / (np.sqrt(N * sparsity) + 1e-6)
            W_A = W_base + 0.1 * rng.standard_normal((N, N)) * (rng.random((N, N)) < sparsity)
            W_B = W_base + 0.1 * rng.standard_normal((N, N)) * (rng.random((N, N)) < sparsity)

            x_A = rng.random(N) * 0.5 + 0.25
            x_B = rng.random(N) * 0.5 + 0.25
            sqrt_dt = np.sqrt(dt)

            for step in range(burn_in + measure_steps):
                activation_A = sigmoid(W_A @ x_A - 0.5)
                x_A = x_A + dt * (-x_A / tau + activation_A) + sqrt_dt * noise * rng.standard_normal(N)
                x_A = np.clip(x_A, 0, 1)

                target = pca_encode_decode(x_A, k)
                activation_B = sigmoid(W_B @ x_B - 0.5)
                code_constraint = lambd * (target - x_B)
                x_B = x_B + dt * (-x_B / tau + activation_B + code_constraint) + sqrt_dt * noise * 0.5 * rng.standard_normal(N)
                x_B = np.clip(x_B, 0, 1)

            trial_A.append(grn_complexity(x_A))
            trial_B.append(grn_complexity(x_B))
            trial_mm.append(np.mean(np.abs(x_A - x_B)))

        results['k'].append(k)
        results['Neff_A'].append(np.mean(trial_A))
        results['Neff_B'].append(np.mean(trial_B))
        results['Neff_A_std'].append(np.std(trial_A) / np.sqrt(n_trials))
        results['Neff_B_std'].append(np.std(trial_B) / np.sqrt(n_trials))
        results['mismatch'].append(np.mean(trial_mm))

        print(f"  k={k:3d}: Neff(A)={results['Neff_A'][-1]:.1f}, Neff(B)={results['Neff_B'][-1]:.1f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(results['k'], results['Neff_A'], yerr=results['Neff_A_std'],
                 marker='o', linestyle='-', color='gray', label=r'System $A$', capsize=3)
    ax1.errorbar(results['k'], results['Neff_B'], yerr=results['Neff_B_std'],
                 marker='s', linestyle='-', color='#2166ac', label=r'System $B$', capsize=3)
    ax1.set_xlabel('Code bandwidth $k$')
    ax1.set_ylabel(r'Effective dimensionality $N_{\mathrm{eff}}$')
    ax1.set_xscale('log', base=2)
    ax1.legend()
    ax1.set_title(f'(A) Gene Regulatory Network, N={N}')
    ax1.grid(True, alpha=0.3)

    ax2.plot(results['k'], results['mismatch'], 's-', color='#b2182b')
    ax2.set_xlabel('Code bandwidth $k$')
    ax2.set_ylabel('Mismatch')
    ax2.set_xscale('log', base=2)
    ax2.set_title('(B) Tracking error')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_supp_grn_tuned.pdf')
    print(f"\nSaved fig_supp_grn_tuned.pdf")


# =============================================================================
# 3. ALTERNATIVE METRICS
# =============================================================================

def run_alternative_metrics():
    """Compute non-Fourier complexity metrics."""
    print("\n" + "="*70)
    print("ALTERNATIVE COMPLEXITY METRICS")
    print("="*70)

    N = 64
    k_values = [1, 2, 4, 8, 16, 32]
    n_trials = 12
    K, lambd, noise, dt = 0.5, 1.0, 0.3, 0.1
    burn_in, measure_steps, traj_len = 500, 300, 100
    rng = np.random.default_rng(42)
    sqrt_dt = np.sqrt(dt)

    def pca_participation(traj):
        centered = traj - traj.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        if len(eigvals) == 0:
            return 1.0
        return np.sum(eigvals)**2 / np.sum(eigvals**2)

    def gradient_energy(theta):
        dtheta = np.diff(theta)
        dtheta = np.mod(dtheta + np.pi, 2*np.pi) - np.pi
        return np.mean(dtheta**2)

    def order_param(theta):
        return np.abs(np.mean(np.exp(1j * theta)))

    results = {'k': [],
               'spectral_B': [], 'pca_B': [], 'gradient_B': [], 'order_B': [],
               'spectral_B_std': [], 'pca_B_std': [], 'gradient_B_std': [], 'order_B_std': []}

    for k in k_values:
        trials = {'spectral': [], 'pca': [], 'gradient': [], 'order': []}

        for trial in range(n_trials):
            theta_A = rng.uniform(-np.pi, np.pi, N)
            theta_B = rng.uniform(-np.pi, np.pi, N)
            omega_A = 0.5 + 0.15 * rng.standard_normal(N)
            omega_B = 0.5 + 0.15 * rng.standard_normal(N)

            # Burn-in
            for _ in range(burn_in):
                theta_A, theta_B = kuramoto_coupled_step(
                    theta_A, theta_B, omega_A, omega_B, k, K, lambd, noise, dt, rng)

            # Measure
            traj_B = np.zeros((traj_len, N))
            spectral_sum, gradient_sum, order_sum = 0.0, 0.0, 0.0

            for t in range(measure_steps):
                theta_A, theta_B = kuramoto_coupled_step(
                    theta_A, theta_B, omega_A, omega_B, k, K, lambd, noise, dt, rng)
                if t < traj_len:
                    traj_B[t] = theta_B
                spectral_sum += spectral_complexity(theta_B)
                gradient_sum += gradient_energy(theta_B)
                order_sum += order_param(theta_B)

            trials['spectral'].append(spectral_sum / measure_steps)
            trials['pca'].append(pca_participation(traj_B))
            trials['gradient'].append(gradient_sum / measure_steps)
            trials['order'].append(order_sum / measure_steps)

        results['k'].append(k)
        for metric in ['spectral', 'pca', 'gradient', 'order']:
            results[f'{metric}_B'].append(np.mean(trials[metric]))
            results[f'{metric}_B_std'].append(np.std(trials[metric]) / np.sqrt(n_trials))

        print(f"  k={k:2d}: Spectral={results['spectral_B'][-1]:.1f}, "
              f"Gradient={results['gradient_B'][-1]:.3f}, Order={results['order_B'][-1]:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [('spectral_B', 'Spectral entropy (Fourier)', r'$N_{\mathrm{eff}}$'),
               ('pca_B', 'PCA participation ratio', r'$D_{\mathrm{eff}}$'),
               ('gradient_B', 'Spatial gradient energy', r'$\langle|\nabla\theta|^2\rangle$'),
               ('order_B', 'Kuramoto order parameter', r'$R$')]

    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        ax.errorbar(results['k'], results[key], yerr=results[f'{key}_std'],
                    marker='s', linestyle='-', color='#2166ac', capsize=3)
        ax.set_xlabel('Code bandwidth $k$')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log', base=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_supp_alternative_metrics.pdf')
    print(f"\nSaved fig_supp_alternative_metrics.pdf")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supplementary simulations')
    parser.add_argument('--all', action='store_true', help='Run all simulations')
    parser.add_argument('--large', action='store_true', help='Large-scale Kuramoto')
    parser.add_argument('--grn', action='store_true', help='Gene regulatory network')
    parser.add_argument('--metrics', action='store_true', help='Alternative metrics')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.all or not any([args.large, args.grn, args.metrics]):
        run_large_scale_kuramoto()
        run_grn()
        run_alternative_metrics()
    else:
        if args.large:
            run_large_scale_kuramoto()
        if args.grn:
            run_grn()
        if args.metrics:
            run_alternative_metrics()

    print("\n" + "="*70)
    print("ALL SUPPLEMENTARY SIMULATIONS COMPLETE")
    print("="*70)
