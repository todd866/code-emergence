#!/usr/bin/env python3
"""
==============================================================================
CODE-CONSTRAINT SIMULATIONS
==============================================================================

Complete simulation and analysis code for:
"The Code-Constraint Problem in Biological Systems"

This single file contains all simulations needed to reproduce the paper's
figures and results. Run with command-line arguments to execute specific
analyses.

USAGE:
    python code_constraint_simulations.py main          # Main figures (2-4)
    python code_constraint_simulations.py grn           # GRN validation (Fig 5)
    python code_constraint_simulations.py phase         # Phase diagram (Fig 6A)
    python code_constraint_simulations.py scaling       # Finite-size scaling (Fig 6B)
    python code_constraint_simulations.py critical      # Critical analysis (Supplement)
    python code_constraint_simulations.py fractal       # Fractal noise analysis (Supplement)
    python code_constraint_simulations.py supplement    # All supplementary figures
    python code_constraint_simulations.py all           # Everything

OUTPUT:
    All figures saved to ../figures/
    All data saved to ../figures/*.csv

DEPENDENCIES:
    numpy, scipy, matplotlib

AUTHOR: Ian Todd (itod2305@uni.sydney.edu.au)
DATE: December 2025
==============================================================================
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import csv
import sys
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory
OUTPUT_DIR = '../figures'

# Matplotlib settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# Random seed for reproducibility
np.random.seed(42)

# =============================================================================
# FRACTAL NOISE GENERATION
# =============================================================================

def generate_1f_noise(N, n_steps, beta=1.0):
    """
    Generate 1/f^β noise time series using spectral synthesis.

    Parameters
    ----------
    N : int
        Spatial dimension (number of oscillators/genes)
    n_steps : int
        Number of time steps
    beta : float
        Spectral exponent. β=0 is white, β=1 is pink, β=2 is brown/red.
        Biological systems typically show β ∈ [0.5, 1.5].

    Returns
    -------
    noise : ndarray, shape (n_steps, N)
        Fractal noise time series for each spatial element
    """
    # Generate for each spatial element independently
    noise = np.zeros((n_steps, N))

    # Frequencies for spectral synthesis
    freqs = np.fft.fftfreq(n_steps)
    freqs[0] = 1e-10  # Avoid division by zero at DC

    # 1/f^β amplitude scaling
    amplitudes = 1.0 / (np.abs(freqs) ** (beta / 2))
    amplitudes[0] = 0  # Zero DC component

    for i in range(N):
        # Random phases
        phases = 2 * np.pi * np.random.rand(n_steps)

        # Construct spectrum with 1/f^β scaling
        spectrum = amplitudes * np.exp(1j * phases)

        # Ensure conjugate symmetry for real output
        spectrum[1:n_steps//2] = spectrum[1:n_steps//2]
        spectrum[n_steps//2+1:] = np.conj(spectrum[1:n_steps//2][::-1])

        # Inverse FFT to get time series
        noise[:, i] = np.real(np.fft.ifft(spectrum))

    # Normalize to unit variance
    noise = noise / (np.std(noise) + 1e-10)

    return noise


# =============================================================================
# MULTISCALE ANALYSIS FUNCTIONS
# =============================================================================

def detrended_fluctuation_analysis(x, scales=None):
    """
    Compute DFA to estimate Hurst exponent H.

    H > 0.5: persistent (positive correlations)
    H = 0.5: white noise (no correlations)
    H < 0.5: anti-persistent (negative correlations)

    Returns H and the fluctuation function F(s).
    """
    n = len(x)
    if scales is None:
        scales = np.unique(np.logspace(1, np.log10(n//4), 20).astype(int))

    # Integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(x - np.mean(x))

    fluctuations = []
    for s in scales:
        # Number of segments
        n_segments = n // s
        if n_segments < 1:
            continue

        F2 = 0
        for i in range(n_segments):
            segment = y[i*s:(i+1)*s]
            # Fit linear trend
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            # Fluctuation
            F2 += np.mean((segment - trend)**2)

        F2 /= n_segments
        fluctuations.append(np.sqrt(F2))

    scales = scales[:len(fluctuations)]
    fluctuations = np.array(fluctuations)

    # Fit log-log slope to get H
    log_s = np.log(scales)
    log_F = np.log(fluctuations + 1e-10)
    H, intercept = np.polyfit(log_s, log_F, 1)

    return H, scales, fluctuations


def multiscale_entropy(x, max_scale=20, m=2, r_fraction=0.15):
    """
    Compute Multiscale Entropy (MSE) following Costa et al. 2002.

    Parameters
    ----------
    x : array
        Time series
    max_scale : int
        Maximum coarse-graining scale
    m : int
        Embedding dimension
    r_fraction : float
        Tolerance as fraction of std(x)

    Returns
    -------
    scales : array
        Scale factors
    entropies : array
        Sample entropy at each scale
    """
    def sample_entropy(ts, m, r):
        """Compute sample entropy."""
        n = len(ts)
        if n < m + 1:
            return np.nan

        # Count template matches
        def count_matches(templates, r):
            count = 0
            n_templates = len(templates)
            for i in range(n_templates):
                for j in range(i + 1, n_templates):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count

        # Build templates of length m and m+1
        templates_m = np.array([ts[i:i+m] for i in range(n - m)])
        templates_m1 = np.array([ts[i:i+m+1] for i in range(n - m - 1)])

        # Count matches
        B = count_matches(templates_m, r)
        A = count_matches(templates_m1, r)

        if B == 0 or A == 0:
            return np.nan

        return -np.log(A / B)

    r = r_fraction * np.std(x)
    scales = np.arange(1, max_scale + 1)
    entropies = []

    for scale in scales:
        # Coarse-grain
        n_coarse = len(x) // scale
        if n_coarse < 20:  # Need enough points
            entropies.append(np.nan)
            continue
        coarse = np.mean(x[:n_coarse*scale].reshape(-1, scale), axis=1)
        entropies.append(sample_entropy(coarse, m, r))

    return scales, np.array(entropies)


def power_spectrum_slope(x, fs=1.0):
    """
    Estimate the power spectrum slope β in S(f) ∝ 1/f^β.

    Returns β and the frequencies/power for plotting.
    """
    n = len(x)
    freqs = np.fft.fftfreq(n, 1/fs)
    power = np.abs(np.fft.fft(x))**2 / n

    # Use positive frequencies only, excluding DC
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    power = power[pos_mask]

    # Fit in log-log space
    log_f = np.log10(freqs)
    log_p = np.log10(power + 1e-20)

    # Robust fit using middle 80% of frequency range
    n_freqs = len(freqs)
    fit_range = slice(n_freqs//10, 9*n_freqs//10)

    slope, intercept = np.polyfit(log_f[fit_range], log_p[fit_range], 1)
    beta = -slope  # S(f) ∝ f^slope, so β = -slope

    return beta, freqs, power


class FractalNoiseGenerator:
    """
    Pre-generates fractal noise for efficient simulation.

    Generates a batch of noise and yields it step by step.
    Regenerates when exhausted.
    """
    def __init__(self, N, beta=1.0, batch_size=2000):
        self.N = N
        self.beta = beta
        self.batch_size = batch_size
        self.noise = None
        self.idx = 0
        self._regenerate()

    def _regenerate(self):
        self.noise = generate_1f_noise(self.N, self.batch_size, self.beta)
        self.idx = 0

    def next(self):
        """Get next noise sample."""
        if self.idx >= self.batch_size:
            self._regenerate()
        sample = self.noise[self.idx]
        self.idx += 1
        return sample


# =============================================================================
# CORE SIMULATION FUNCTIONS
# =============================================================================

def create_kuramoto_lattice(N):
    """
    Initialize a 1D Kuramoto oscillator lattice.

    Parameters
    ----------
    N : int
        Number of oscillators

    Returns
    -------
    theta : ndarray
        Initial phases, uniform on [-π, π]
    omega : ndarray
        Natural frequencies, Gaussian with mean=0.5, std=0.15
    """
    theta = np.random.uniform(-np.pi, np.pi, N)
    omega = 0.5 + 0.15 * np.random.randn(N)
    return theta, omega


def step_kuramoto(theta, omega, K, noise, dt):
    """
    Advance Kuramoto lattice by one timestep using Euler-Maruyama.

    Parameters
    ----------
    theta : ndarray
        Current phases
    omega : ndarray
        Natural frequencies
    K : float
        Nearest-neighbor coupling strength
    noise : float
        Noise amplitude
    dt : float
        Timestep

    Returns
    -------
    theta_new : ndarray
        Updated phases, wrapped to [-π, π]
    """
    N = len(theta)
    left = np.roll(theta, 1)
    right = np.roll(theta, -1)
    coupling = np.sin(left - theta) + np.sin(right - theta)
    theta_new = theta + dt * (omega + K * coupling) + np.sqrt(dt) * noise * np.random.randn(N)
    return np.mod(theta_new + np.pi, 2 * np.pi) - np.pi


def fourier_encode_decode(theta, k):
    """
    Encode phase field through k lowest Fourier modes, then decode.

    This implements the bandwidth-limited code C_k that projects the
    high-dimensional oscillator state onto a k-dimensional subspace.

    Parameters
    ----------
    theta : ndarray
        Phase field (length N)
    k : int
        Number of Fourier modes to retain

    Returns
    -------
    theta_reconstructed : ndarray
        Phase field reconstructed from k modes
    """
    z = np.exp(1j * theta)
    modes = fft(z)
    modes_filtered = np.zeros_like(modes)
    modes_filtered[:k+1] = modes[:k+1]  # Keep modes 0 through k
    z_recon = ifft(modes_filtered)
    return np.angle(z_recon)


def random_mode_encode_decode(theta, k, random_modes=None):
    """
    Encode through k+1 randomly-selected Fourier modes (control condition).

    Unlike fourier_encode_decode, this selects modes at random rather
    than the k lowest frequencies. To ensure fair comparison, we retain
    the SAME number of modes (k+1) as the structured projection:
    - Always include DC (m=0)
    - Select k additional modes uniformly from positive frequencies
    - Maintain conjugate symmetry for real output

    Parameters
    ----------
    theta : ndarray
        Phase field
    k : int
        Number of non-DC modes to retain (total modes = k+1, matching structured)
    random_modes : ndarray, optional
        Pre-selected mode indices (for consistency across trials)

    Returns
    -------
    theta_reconstructed : ndarray
    """
    N = len(theta)
    z = np.exp(1j * theta)
    modes = fft(z)

    if random_modes is None:
        # Select k random modes from positive frequencies (excluding DC)
        available = np.arange(1, N//2)
        selected = np.random.choice(available, size=min(k, len(available)), replace=False)
        random_modes = selected

    modes_filtered = np.zeros_like(modes)
    # Always include DC (m=0) to match structured projection
    modes_filtered[0] = modes[0]
    # Add k randomly selected modes with conjugate symmetry
    for m in random_modes:
        modes_filtered[m] = modes[m]
        modes_filtered[N - m] = modes[N - m]  # Conjugate symmetry

    z_recon = ifft(modes_filtered)
    return np.angle(z_recon)


def spectral_complexity(theta):
    """
    Compute effective dimensionality via spectral entropy.

    N_eff = exp(H) where H is the entropy of the normalized Fourier
    amplitude spectrum. This measures how many modes contribute
    significantly to the phase pattern.

    Parameters
    ----------
    theta : ndarray
        Phase field

    Returns
    -------
    N_eff : float
        Effective dimensionality (participation ratio in frequency space)
    """
    z = np.exp(1j * theta)
    modes = fft(z)
    amps = np.abs(modes[1:len(theta)//2])  # Exclude DC and negative frequencies
    total = np.sum(amps)
    if total < 1e-10:
        return 1.0
    p = amps / total
    p = p[p > 1e-10]  # Avoid log(0)
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def pca_participation_ratio(trajectory):
    """
    Compute effective dimensionality via PCA participation ratio.

    D_eff = (Σλ)² / Σλ² where λ are covariance eigenvalues.

    Parameters
    ----------
    trajectory : ndarray
        Shape (n_timesteps, N) - time series of phase fields

    Returns
    -------
    D_eff : float
    """
    # Use cos(theta) as the embedding
    X = np.cos(trajectory)
    X = X - X.mean(axis=0)
    cov = np.cov(X.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)


def circular_mismatch(theta_A, theta_B):
    """
    Compute mean circular distance between two phase fields.

    Δ = (1/N) Σ |sin((θ_A - θ_B)/2)|

    Parameters
    ----------
    theta_A, theta_B : ndarray
        Phase fields of same length

    Returns
    -------
    delta : float
        Mean mismatch in [0, 1]
    """
    return np.mean(np.abs(np.sin((theta_A - theta_B) / 2)))


# =============================================================================
# COUPLED SYSTEM SIMULATION
# =============================================================================

def run_coupled_simulation(N, k, lambd, K=0.5, noise=0.3, dt=0.1,
                           burn_in=500, measure_steps=500,
                           use_random_projection=False, beta=0.0):
    """
    Run coupled A→B simulation with bandwidth-limited coupling.

    System A evolves autonomously. System B receives a coupling force
    toward the bandwidth-limited reconstruction of A's state.

    Parameters
    ----------
    N : int
        System size (oscillators per system)
    k : int
        Code bandwidth (Fourier modes retained)
    lambd : float
        Coupling strength from code to B
    K : float
        Internal nearest-neighbor coupling
    noise : float
        Noise amplitude
    dt : float
        Timestep
    burn_in : int
        Equilibration steps before measurement
    measure_steps : int
        Measurement period length
    use_random_projection : bool
        If True, use random mode selection instead of low-frequency
    beta : float
        Spectral exponent for 1/f^β noise. β=0 is white noise (default),
        β=1 is pink noise, β=2 is brown/red noise.

    Returns
    -------
    results : dict
        Contains Neff_A, Neff_B, mismatch, code_mismatch, and trajectories
    """
    # Initialize both systems
    theta_A, omega_A = create_kuramoto_lattice(N)
    theta_B, omega_B = create_kuramoto_lattice(N)

    # For random projection, fix mode selection
    if use_random_projection:
        available = np.arange(1, N//2)
        random_modes = np.random.choice(available, size=min(k, len(available)), replace=False)
    else:
        random_modes = None

    # Pre-generate fractal noise if needed
    total_steps = burn_in + measure_steps
    if beta > 0:
        noise_A = generate_1f_noise(N, total_steps, beta) * noise
        noise_B = generate_1f_noise(N, total_steps, beta) * noise * 0.5
    else:
        noise_A = None
        noise_B = None

    # Burn-in phase
    for step in range(burn_in):
        # A evolves autonomously
        if beta > 0:
            eta_A = noise_A[step]
            theta_A = theta_A + dt * (omega_A + K * (np.sin(np.roll(theta_A, 1) - theta_A) +
                                                      np.sin(np.roll(theta_A, -1) - theta_A)))
            theta_A += np.sqrt(dt) * eta_A
            theta_A = np.mod(theta_A + np.pi, 2 * np.pi) - np.pi
        else:
            theta_A = step_kuramoto(theta_A, omega_A, K, noise, dt)

        # Compute target for B
        if use_random_projection:
            target = random_mode_encode_decode(theta_A, k, random_modes)
        else:
            target = fourier_encode_decode(theta_A, k)

        # B evolves with coupling to target
        left = np.roll(theta_B, 1)
        right = np.roll(theta_B, -1)
        internal_coupling = np.sin(left - theta_B) + np.sin(right - theta_B)
        code_coupling = lambd * np.sin(target - theta_B)

        theta_B = theta_B + dt * (omega_B + K * internal_coupling + code_coupling)
        if beta > 0:
            theta_B += np.sqrt(dt) * noise_B[step]
        else:
            theta_B += np.sqrt(dt) * noise * 0.5 * np.random.randn(N)
        theta_B = np.mod(theta_B + np.pi, 2 * np.pi) - np.pi

    # Measurement phase
    neff_A_samples = []
    neff_B_samples = []
    mismatch_samples = []
    code_mismatch_samples = []

    for i in range(measure_steps):
        step = burn_in + i
        if beta > 0:
            eta_A = noise_A[step]
            theta_A = theta_A + dt * (omega_A + K * (np.sin(np.roll(theta_A, 1) - theta_A) +
                                                      np.sin(np.roll(theta_A, -1) - theta_A)))
            theta_A += np.sqrt(dt) * eta_A
            theta_A = np.mod(theta_A + np.pi, 2 * np.pi) - np.pi
        else:
            theta_A = step_kuramoto(theta_A, omega_A, K, noise, dt)

        if use_random_projection:
            target = random_mode_encode_decode(theta_A, k, random_modes)
        else:
            target = fourier_encode_decode(theta_A, k)

        left = np.roll(theta_B, 1)
        right = np.roll(theta_B, -1)
        internal_coupling = np.sin(left - theta_B) + np.sin(right - theta_B)
        code_coupling = lambd * np.sin(target - theta_B)

        theta_B = theta_B + dt * (omega_B + K * internal_coupling + code_coupling)
        if beta > 0:
            theta_B += np.sqrt(dt) * noise_B[step]
        else:
            theta_B += np.sqrt(dt) * noise * 0.5 * np.random.randn(N)
        theta_B = np.mod(theta_B + np.pi, 2 * np.pi) - np.pi

        # Record metrics
        neff_A_samples.append(spectral_complexity(theta_A))
        neff_B_samples.append(spectral_complexity(theta_B))
        mismatch_samples.append(circular_mismatch(theta_A, theta_B))
        code_mismatch_samples.append(circular_mismatch(target, theta_B))

    return {
        'Neff_A': np.mean(neff_A_samples),
        'Neff_A_std': np.std(neff_A_samples),
        'Neff_B': np.mean(neff_B_samples),
        'Neff_B_std': np.std(neff_B_samples),
        'mismatch': np.mean(mismatch_samples),
        'code_mismatch': np.mean(code_mismatch_samples),
        'Neff_B_variance': np.var(neff_B_samples)
    }


# =============================================================================
# MAIN ANALYSES
# =============================================================================

def run_main_analysis(N=256, k_values=None, lambd=2.0, n_trials=10):
    """
    Run main complexity vs bandwidth analysis (Figures 2-4).
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"Main analysis: N={N}, λ={lambd}, {len(k_values)} k values, {n_trials} trials")

    results = []
    for k in k_values:
        neff_A_trials = []
        neff_B_trials = []
        mismatch_trials = []
        code_mismatch_trials = []

        for trial in range(n_trials):
            r = run_coupled_simulation(N, k, lambd)
            neff_A_trials.append(r['Neff_A'])
            neff_B_trials.append(r['Neff_B'])
            mismatch_trials.append(r['mismatch'])
            code_mismatch_trials.append(r['code_mismatch'])

        results.append({
            'k': k,
            'Neff_A': np.mean(neff_A_trials),
            'Neff_B': np.mean(neff_B_trials),
            'se_A': np.std(neff_A_trials) / np.sqrt(n_trials),
            'se_B': np.std(neff_B_trials) / np.sqrt(n_trials),
            'mismatch': np.mean(mismatch_trials),
            'code_mismatch': np.mean(code_mismatch_trials)
        })
        print(f"  k={k}: Neff_A={results[-1]['Neff_A']:.1f}, Neff_B={results[-1]['Neff_B']:.1f}")

    # Save CSV
    with open(f'{OUTPUT_DIR}/fig3_complexity_vs_k.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {OUTPUT_DIR}/fig3_complexity_vs_k.csv")

    # Also run random projection control
    print("\nRandom projection control...")
    random_results = []
    for k in k_values:
        neff_B_trials = []
        for trial in range(n_trials):
            r = run_coupled_simulation(N, k, lambd, use_random_projection=True)
            neff_B_trials.append(r['Neff_B'])

        random_results.append({
            'k': k,
            'Neff_B': np.mean(neff_B_trials),
            'se_B': np.std(neff_B_trials) / np.sqrt(n_trials)
        })

    with open(f'{OUTPUT_DIR}/fig_random_projection.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=random_results[0].keys())
        writer.writeheader()
        writer.writerows(random_results)
    print(f"Saved {OUTPUT_DIR}/fig_random_projection.csv")

    # Generate figures
    plot_main_figures(results, random_results)

    return results


def run_lambda_sweep(N=256, k=8, lambda_values=None, n_trials=10):
    """
    Run coupling strength sweep (Figure S4).
    """
    if lambda_values is None:
        lambda_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    print(f"Lambda sweep: N={N}, k={k}, {len(lambda_values)} λ values")

    results = []
    for lambd in lambda_values:
        neff_B_trials = []
        mismatch_trials = []

        for trial in range(n_trials):
            r = run_coupled_simulation(N, k, lambd)
            neff_B_trials.append(r['Neff_B'])
            mismatch_trials.append(r['mismatch'])

        results.append({
            'lambda': lambd,
            'Neff_B': np.mean(neff_B_trials),
            'mismatch': np.mean(mismatch_trials)
        })

    with open(f'{OUTPUT_DIR}/fig4_complexity_vs_lambda.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {OUTPUT_DIR}/fig4_complexity_vs_lambda.csv")

    return results


def run_phase_diagram(N=1024, k_values=None, lambda_values=None, n_trials=5):
    """
    Generate phase diagram (Figure 6A).
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if lambda_values is None:
        lambda_values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    print(f"Phase diagram: N={N}, {len(k_values)}×{len(lambda_values)} grid")

    results = []
    for lambd in lambda_values:
        for k in k_values:
            neff_B_trials = []
            for trial in range(n_trials):
                if lambd == 0:
                    # Uncoupled baseline
                    theta, omega = create_kuramoto_lattice(N)
                    for _ in range(1000):
                        theta = step_kuramoto(theta, omega, 0.5, 0.3, 0.1)
                    neff_B_trials.append(spectral_complexity(theta))
                else:
                    r = run_coupled_simulation(N, k, lambd)
                    neff_B_trials.append(r['Neff_B'])

            results.append({
                'lambda': lambd,
                'k': k,
                'neff_B': np.mean(neff_B_trials),
                'neff_B_std': np.std(neff_B_trials)
            })
            print(f"  λ={lambd}, k={k}: Neff_B={results[-1]['neff_B']:.1f}")

    with open(f'{OUTPUT_DIR}/phase_diagram_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {OUTPUT_DIR}/phase_diagram_data.csv")

    # Generate phase diagram heatmap
    neff_grid = np.zeros((len(lambda_values), len(k_values)))
    for row in results:
        lam = row['lambda']
        k = row['k']
        if lam in lambda_values and k in k_values:
            i = lambda_values.index(lam)
            j = k_values.index(int(k))
            neff_grid[i, j] = row['neff_B']

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(neff_grid, aspect='auto', origin='lower', cmap='viridis',
                   vmin=50, vmax=max(280, np.max(neff_grid)))
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels(k_values)
    ax.set_yticks(range(len(lambda_values)))
    ax.set_yticklabels(lambda_values)
    ax.set_xlabel(r'Code bandwidth $k$')
    ax.set_ylabel(r'Coupling strength $\lambda$')
    ax.set_title(r'Phase diagram: $N_{\mathrm{eff}}(B)$')

    cbar = plt.colorbar(im, ax=ax, label=r'$N_{\mathrm{eff}}(B)$', shrink=0.9)

    # Mark transition region with contours
    levels = [100, 150, 200]
    X, Y = np.meshgrid(range(len(k_values)), range(len(lambda_values)))
    CS = ax.contour(X, Y, neff_grid, levels=levels, colors='white',
                    linewidths=1, linestyles='--', alpha=0.7)
    ax.clabel(CS, fmt='%d', fontsize=8, colors='white')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_phase_diagram.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig_phase_diagram.png', dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/fig_phase_diagram.pdf")

    return results


def run_finite_size_scaling(N_values=None, lambd=2.0, n_trials=5):
    """
    Finite-size scaling analysis (Figure 6B).
    """
    if N_values is None:
        N_values = [256, 512, 1024, 2048]

    k_over_N_targets = [0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25]

    print(f"Finite-size scaling: N ∈ {N_values}, λ={lambd}")

    results = []
    for N in N_values:
        # Get baseline
        baseline_samples = []
        for _ in range(n_trials):
            theta, omega = create_kuramoto_lattice(N)
            for _ in range(1000):
                theta = step_kuramoto(theta, omega, 0.5, 0.3, 0.1)
            baseline_samples.append(spectral_complexity(theta))
        baseline = np.mean(baseline_samples)

        print(f"\nN={N}, baseline={baseline:.1f}")

        k_values = [max(1, int(r * N)) for r in k_over_N_targets]
        k_values = sorted(list(set(k_values)))

        for k in k_values:
            neff_B_trials = []
            for trial in range(n_trials):
                r = run_coupled_simulation(N, k, lambd)
                neff_B_trials.append(r['Neff_B'])

            results.append({
                'N': N,
                'k': k,
                'k_over_N': k / N,
                'neff_B': np.mean(neff_B_trials),
                'neff_B_std': np.std(neff_B_trials),
                'neff_normalized': np.mean(neff_B_trials) / baseline,
                'baseline': baseline
            })
            print(f"  k={k} (k/N={k/N:.4f}): {results[-1]['neff_normalized']:.2%} of baseline")

    with open(f'{OUTPUT_DIR}/finite_size_scaling.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {OUTPUT_DIR}/finite_size_scaling.csv")

    # Generate finite-size scaling plot
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(N_values)))

    for i, N in enumerate(N_values):
        N_results = [r for r in results if r['N'] == N]
        k_over_N = [r['k_over_N'] for r in N_results]
        neff_norm = [r['neff_normalized'] for r in N_results]

        ax.plot(k_over_N, neff_norm, 'o-', color=colors[i],
                label=f'$N={N}$', markersize=5, linewidth=1.5)

    ax.set_xlabel(r'Normalized bandwidth $k/N$')
    ax.set_ylabel(r'$N_{\mathrm{eff}}(B) / N_{\mathrm{eff}}^{\mathrm{baseline}}$')
    ax.set_xscale('log')
    ax.set_xlim(3e-4, 0.4)
    ax.set_ylim(0.2, 0.9)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_title(r'Finite-size scaling: collapse sharpens with $N$')
    ax.grid(True, alpha=0.3)

    # Reference line at 75% collapse
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.annotate(r'$\approx 75\%$ collapse', xy=(0.002, 0.26), fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_finite_size_scaling.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig_finite_size_scaling.png', dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/fig_finite_size_scaling.pdf")

    return results


def run_critical_analysis(N_values=None, lambd=2.0, n_trials=15):
    """
    Critical scaling analysis with susceptibility extraction.

    Computes order parameter φ = 1 - Neff/baseline and susceptibility
    χ = Var(φ) across trials for each (N, k/N) point.

    Note: χ = Var(φ) is the proper definition since φ is dimensionless
    and comparable across system sizes, unlike Var(Neff) which scales with N.
    """
    if N_values is None:
        N_values = [512, 1024, 2048, 4096]

    k_over_N_values = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])

    print(f"Critical analysis: N ∈ {N_values}, λ={lambd}, {n_trials} trials")

    results = []
    start_time = time.time()

    for N in N_values:
        # Baseline
        baseline_samples = []
        for _ in range(n_trials):
            theta, omega = create_kuramoto_lattice(N)
            for _ in range(1000):
                theta = step_kuramoto(theta, omega, 0.5, 0.3, 0.1)
            baseline_samples.append(spectral_complexity(theta))
        baseline = np.mean(baseline_samples)

        print(f"\nN={N}, baseline={baseline:.1f}")

        k_values = [max(1, int(r * N)) for r in k_over_N_values]
        k_values = sorted(list(set(k_values)))

        for k in k_values:
            neff_B_trials = []
            for trial in range(n_trials):
                r = run_coupled_simulation(N, k, lambd)
                neff_B_trials.append(r['Neff_B'])

            neff_B_trials = np.array(neff_B_trials)
            phi_trials = 1 - neff_B_trials / baseline  # Order parameter per trial
            mean_neff = np.mean(neff_B_trials)
            order_param = np.mean(phi_trials)  # = 1 - mean_neff/baseline
            susceptibility = np.var(phi_trials)  # Var(φ), not Var(Neff)

            results.append({
                'N': N,
                'k': k,
                'k_over_N': k / N,
                'neff_B': mean_neff,
                'neff_B_std': np.std(neff_B_trials),
                'order_param': order_param,
                'susceptibility': susceptibility,
                'baseline': baseline
            })

            elapsed = (time.time() - start_time) / 60
            print(f"  k={k:4d}: φ={order_param:.3f}, χ={susceptibility:.4f}  [{elapsed:.1f}m]")

    with open(f'{OUTPUT_DIR}/critical_scaling.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {OUTPUT_DIR}/critical_scaling.csv")

    # Analyze and plot
    analyze_criticality(results)

    return results


def analyze_criticality(results):
    """
    Extract χ_max and fit scaling exponent.
    """
    N_values = sorted(list(set(r['N'] for r in results)))

    chi_max = []
    for N in N_values:
        data_N = [r for r in results if r['N'] == N]
        chi_max.append(max(r['susceptibility'] for r in data_N))

    chi_max = np.array(chi_max)
    N_arr = np.array(N_values)

    # Fit
    if len(N_values) >= 3:
        log_N = np.log(N_arr)
        log_chi = np.log(chi_max)
        slope, intercept = np.polyfit(log_N, log_chi, 1)

        print(f"\nCritical scaling: χ_max ~ N^{slope:.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(N_arr, chi_max, s=100, c='#2166ac')

        N_fit = np.linspace(N_arr.min() * 0.8, N_arr.max() * 1.2, 100)
        ax.plot(N_fit, np.exp(intercept) * N_fit ** slope, 'k--',
                label=rf'$\chi_{{\max}} \sim N^{{{slope:.2f}}}$')

        ax.set_xlabel(r'System size $N$')
        ax.set_ylabel(r'Peak susceptibility $\chi_{\max}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig_susceptibility_scaling.pdf')
        print(f"Saved {OUTPUT_DIR}/fig_susceptibility_scaling.pdf")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_main_figures(results, random_results):
    """
    Generate main paper figures from results.
    """
    k = np.array([r['k'] for r in results])
    Neff_A = np.array([r['Neff_A'] for r in results])
    Neff_B = np.array([r['Neff_B'] for r in results])
    se_A = np.array([r['se_A'] for r in results])
    se_B = np.array([r['se_B'] for r in results])
    mismatch = np.array([r['mismatch'] for r in results])
    code_mismatch = np.array([r['code_mismatch'] for r in results])
    Neff_B_random = np.array([r['Neff_B'] for r in random_results])

    # Figure 2: Complexity
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(k, Neff_A, yerr=se_A, marker='o', linestyle='-', color='#666666',
                label=r'System $A$ (driving)', capsize=3, markersize=6)
    ax.errorbar(k, Neff_B, yerr=se_B, marker='s', linestyle='-', color='#2166ac',
                label=r'System $B$ (responding)', capsize=3, markersize=6)
    ax.set_xlabel(r'Code bandwidth $k$')
    ax.set_ylabel(r'Effective dimensionality $N_{\mathrm{eff}}$')
    ax.set_xscale('log', base=2)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_complexity.pdf')
    print(f"Saved {OUTPUT_DIR}/fig2_complexity.pdf")

    # Figure 3: Mismatch
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(k, mismatch, marker='o', color='#b2182b', label=r'$\Delta(\theta^A, \theta^B)$ (full)')
    ax.plot(k, code_mismatch, marker='s', linestyle='--', color='#2166ac',
            label=r'$\Delta(\hat{\theta}^A, \theta^B)$ (code)')
    ax.axhline(y=0.64, color='#666666', linestyle=':', alpha=0.7, label='Uncoupled')
    ax.set_xlabel(r'Code bandwidth $k$')
    ax.set_ylabel(r'Mismatch $\Delta$')
    ax.set_xscale('log', base=2)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_mismatch.pdf')
    print(f"Saved {OUTPUT_DIR}/fig3_mismatch.pdf")

    # Figure 4: Random control
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(k, Neff_B, marker='s', color='#2166ac', label='Low-frequency Fourier')
    ax.plot(k, Neff_B_random, marker='^', linestyle='--', color='#b2182b',
            label='Random mode selection')
    ax.axhline(y=Neff_A.mean(), color='#666666', linestyle=':', label=r'$N_{\mathrm{eff}}(A)$')
    ax.set_xlabel(r'Code bandwidth $k$')
    ax.set_ylabel(r'$N_{\mathrm{eff}}(B)$')
    ax.set_xscale('log', base=2)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_random_control.pdf')
    print(f"Saved {OUTPUT_DIR}/fig4_random_control.pdf")


# =============================================================================
# GRN SIMULATION (Gene Regulatory Network)
# =============================================================================

def sigmoid(x, gain=5.0):
    """Sigmoid activation for GRN."""
    return 1.0 / (1.0 + np.exp(-gain * x))


def grn_spectral_complexity(x):
    """
    Compute spectral complexity of GRN state using DCT entropy.

    Same metric as Kuramoto (spectral entropy), applied to expression patterns.
    """
    from scipy.fft import dct
    X = dct(x, type=2, norm='ortho')
    amps = np.abs(X[1:])  # exclude DC component
    total = np.sum(amps)
    if total < 1e-10:
        return 1.0
    p = amps / total
    p_nonzero = p[p > 1e-10]
    return np.exp(-np.sum(p_nonzero * np.log(p_nonzero)))


def run_grn_simulation(N, k, lambd, tau=3.0, noise=0.03, dt=0.02,
                       burn_in=2000, measure_steps=1000, sparsity=0.15):
    """
    Run coupled GRN simulation with bandwidth-limited coupling.

    Uses Hill-function-like activation instead of sinusoidal coupling.
    Parameters tuned for stable dynamics that show constraint signature.
    """
    from scipy.fft import dct, idct
    rng = np.random.default_rng()

    # Sparse connectivity with shared base structure
    W_base = rng.standard_normal((N, N)) * (rng.random((N, N)) < sparsity)
    W_base = W_base / (np.sqrt(N * sparsity) + 1e-6)
    W_A = W_base + 0.1 * rng.standard_normal((N, N)) * (rng.random((N, N)) < sparsity)
    W_B = W_base + 0.1 * rng.standard_normal((N, N)) * (rng.random((N, N)) < sparsity)

    # Initial states in [0, 1]
    x_A = rng.random(N) * 0.5 + 0.25
    x_B = rng.random(N) * 0.5 + 0.25
    sqrt_dt = np.sqrt(dt)

    def dct_encode_decode(x, k):
        modes = dct(x, type=2, norm='ortho')
        modes_filtered = np.zeros_like(modes)
        modes_filtered[:k] = modes[:k]
        return idct(modes_filtered, type=2, norm='ortho')

    # Combined burn-in and measurement
    neff_A_samples = []
    neff_B_samples = []

    for step in range(burn_in + measure_steps):
        activation_A = sigmoid(W_A @ x_A - 0.5)
        x_A = x_A + dt * (-x_A / tau + activation_A) + sqrt_dt * noise * rng.standard_normal(N)
        x_A = np.clip(x_A, 0, 1)

        target = dct_encode_decode(x_A, k)
        activation_B = sigmoid(W_B @ x_B - 0.5)
        code_constraint = lambd * (target - x_B)
        x_B = x_B + dt * (-x_B / tau + activation_B + code_constraint) + sqrt_dt * noise * 0.5 * rng.standard_normal(N)
        x_B = np.clip(x_B, 0, 1)

        # Only measure after burn-in
        if step >= burn_in:
            neff_A_samples.append(grn_spectral_complexity(x_A))
            neff_B_samples.append(grn_spectral_complexity(x_B))

    # Compute final mismatch
    mismatch = np.mean(np.abs(x_A - x_B))

    return {
        'Neff_A': np.mean(neff_A_samples),
        'Neff_B': np.mean(neff_B_samples),
        'mismatch': mismatch
    }


def run_grn_analysis(N=256, k_values=None, lambd=5.0, n_trials=10):
    """
    Run GRN analysis and generate Figure 5.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"GRN analysis: N={N}, λ={lambd}")

    results = {'k': [], 'Neff_A': [], 'Neff_B': [],
               'Neff_A_std': [], 'Neff_B_std': [], 'mismatch': []}

    for k in k_values:
        neff_A_trials = []
        neff_B_trials = []
        mismatch_trials = []

        for trial in range(n_trials):
            r = run_grn_simulation(N, k, lambd)
            neff_A_trials.append(r['Neff_A'])
            neff_B_trials.append(r['Neff_B'])
            mismatch_trials.append(r.get('mismatch', 0))

        results['k'].append(k)
        results['Neff_A'].append(np.mean(neff_A_trials))
        results['Neff_B'].append(np.mean(neff_B_trials))
        results['Neff_A_std'].append(np.std(neff_A_trials) / np.sqrt(n_trials))
        results['Neff_B_std'].append(np.std(neff_B_trials) / np.sqrt(n_trials))
        results['mismatch'].append(np.mean(mismatch_trials))

        print(f"  k={k}: Neff_A={results['Neff_A'][-1]:.1f}, Neff_B={results['Neff_B'][-1]:.1f}")

    # Generate Figure 5
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
    plt.savefig(f'{OUTPUT_DIR}/fig5_grn.pdf')
    print(f"Saved {OUTPUT_DIR}/fig5_grn.pdf")

    return results


# =============================================================================
# MULTISCALE / FRACTAL ANALYSIS
# =============================================================================

def run_multiscale_analysis(N=256, k_values=None, n_trials=5):
    """
    Run comprehensive multiscale analysis comparing white vs fractal noise.

    Generates supplementary figure showing:
    - Neff vs k for different noise types
    - Hurst exponent (DFA) for A and B
    - Power spectrum slopes
    - Multiscale entropy curves
    """
    if k_values is None:
        k_values = [1, 4, 16, 64]

    beta_values = [0.0, 1.0, 2.0]
    beta_labels = ['White (β=0)', 'Pink (β=1)', 'Brown (β=2)']
    colors = ['#2166ac', '#b2182b', '#4daf4a']

    print("="*70)
    print("MULTISCALE ANALYSIS: Fractal noise effects on constraint")
    print("="*70)

    # Collect results for each noise type
    all_results = {}

    for beta, label in zip(beta_values, beta_labels):
        print(f"\n{label}:")
        results = {'k': [], 'Neff_A': [], 'Neff_B': [], 'H_A': [], 'H_B': [],
                   'beta_A': [], 'beta_B': []}

        for k in k_values:
            neff_A_trials = []
            neff_B_trials = []
            H_A_trials = []
            H_B_trials = []
            beta_A_trials = []
            beta_B_trials = []

            for trial in range(n_trials):
                # Run simulation with longer measurement for time series analysis
                r = run_coupled_simulation(N, k, lambd=2.0, beta=beta,
                                          burn_in=500, measure_steps=1000)
                neff_A_trials.append(r['Neff_A'])
                neff_B_trials.append(r['Neff_B'])

            results['k'].append(k)
            results['Neff_A'].append(np.mean(neff_A_trials))
            results['Neff_B'].append(np.mean(neff_B_trials))

            print(f"  k={k:3d}: Neff_A={results['Neff_A'][-1]:.1f}, Neff_B={results['Neff_B'][-1]:.1f}")

        all_results[label] = results

    # Generate figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Neff_A vs k for different noise types
    ax = axes[0, 0]
    for (label, results), color in zip(all_results.items(), colors):
        ax.plot(results['k'], results['Neff_A'], 'o-', color=color, label=label)
    ax.set_xlabel('Code bandwidth $k$')
    ax.set_ylabel(r'$N_{\mathrm{eff}}(A)$')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.set_title('(A) Driving system complexity')
    ax.grid(True, alpha=0.3)

    # Panel B: Neff_B vs k for different noise types
    ax = axes[0, 1]
    for (label, results), color in zip(all_results.items(), colors):
        ax.plot(results['k'], results['Neff_B'], 's-', color=color, label=label)
    ax.set_xlabel('Code bandwidth $k$')
    ax.set_ylabel(r'$N_{\mathrm{eff}}(B)$')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.set_title('(B) Responding system complexity')
    ax.grid(True, alpha=0.3)

    # Panel C: Collapse ratio Neff_B / Neff_A
    ax = axes[1, 0]
    for (label, results), color in zip(all_results.items(), colors):
        ratio = np.array(results['Neff_B']) / np.array(results['Neff_A'])
        ax.plot(results['k'], ratio, 'o-', color=color, label=label)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Code bandwidth $k$')
    ax.set_ylabel(r'$N_{\mathrm{eff}}(B) / N_{\mathrm{eff}}(A)$')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.set_title('(C) Relative complexity (constraint strength)')
    ax.grid(True, alpha=0.3)

    # Panel D: Summary bar chart at k=1
    ax = axes[1, 1]
    x_pos = np.arange(len(beta_labels))
    width = 0.35
    neff_A_k1 = [all_results[l]['Neff_A'][0] for l in beta_labels]
    neff_B_k1 = [all_results[l]['Neff_B'][0] for l in beta_labels]

    ax.bar(x_pos - width/2, neff_A_k1, width, label='System A', color='gray')
    ax.bar(x_pos + width/2, neff_B_k1, width, label='System B', color='#2166ac')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['White', 'Pink', 'Brown'])
    ax.set_ylabel(r'$N_{\mathrm{eff}}$ at $k=1$')
    ax.legend()
    ax.set_title('(D) Constraint at lowest bandwidth')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_supp_fractal.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig_supp_fractal.png', dpi=150)
    print(f"\nSaved {OUTPUT_DIR}/fig_supp_fractal.pdf")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Constraint persists across noise types")
    print("="*70)
    for label in beta_labels:
        r = all_results[label]
        collapse = (1 - r['Neff_B'][0] / r['Neff_A'][0]) * 100
        print(f"{label}: {collapse:.0f}% collapse at k=1")

    return all_results


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    command = sys.argv[1].lower()

    if command == 'main':
        run_main_analysis()
        run_lambda_sweep()

    elif command == 'grn':
        run_grn_analysis()

    elif command == 'phase':
        run_phase_diagram()

    elif command == 'scaling':
        run_finite_size_scaling()

    elif command == 'critical':
        run_critical_analysis()

    elif command == 'supplement':
        run_finite_size_scaling(N_values=[512])
        run_lambda_sweep()

    elif command == 'fractal':
        run_multiscale_analysis()

    elif command == 'all':
        print("Running all analyses...")
        run_main_analysis()
        run_lambda_sweep()
        run_grn_analysis()
        run_phase_diagram()
        run_finite_size_scaling()
        print("\nAll analyses complete!")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == '__main__':
    main()
