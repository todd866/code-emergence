#!/usr/bin/env python3
"""
Create visualization comparing structured (low-frequency) vs random mode reconstruction.
Shows why mode STRUCTURE matters, not just dimensionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use LaTeX-compatible fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

N = 64
k = 4  # Number of modes to keep

# Create a realistic "wave" pattern (sum of low-frequency modes + noise)
np.random.seed(42)
x = np.linspace(0, 2*np.pi, N)

# Original signal: low-freq structure + high-freq noise
original = (1.5 * np.sin(x) +
            0.8 * np.sin(2*x + 0.5) +
            0.4 * np.sin(3*x + 1.0) +
            0.3 * np.random.randn(N))  # noise

# Compute full FFT
fft_full = np.fft.fft(original)

# Low-frequency reconstruction (modes 0 to k-1)
fft_lowfreq = np.zeros_like(fft_full)
fft_lowfreq[:k] = fft_full[:k]
fft_lowfreq[-k+1:] = fft_full[-k+1:]  # conjugate modes
recon_lowfreq = np.real(np.fft.ifft(fft_lowfreq))

# Random mode reconstruction (k random modes)
np.random.seed(123)
all_modes = list(range(1, N//2))
np.random.shuffle(all_modes)
random_modes = all_modes[:k]

fft_random = np.zeros_like(fft_full)
fft_random[0] = fft_full[0]  # keep DC
for m in random_modes:
    fft_random[m] = fft_full[m]
    fft_random[N-m] = fft_full[N-m]  # conjugate
recon_random = np.real(np.fft.ifft(fft_random))

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))

# Panel A: Original
axes[0].plot(x, original, 'k-', linewidth=1.5)
axes[0].set_xlabel('Position')
axes[0].set_ylabel('Phase')
axes[0].set_title('(A) Original signal', fontsize=11)
axes[0].set_ylim(-3, 3)
axes[0].grid(True, alpha=0.3)

# Panel B: Low-frequency reconstruction
axes[1].plot(x, original, 'k-', linewidth=0.8, alpha=0.3, label='Original')
axes[1].plot(x, recon_lowfreq, 'b-', linewidth=2, label=f'Low-freq ({k} modes)')
axes[1].set_xlabel('Position')
axes[1].set_title(f'(B) Low-frequency code ($k={k}$)', fontsize=11)
axes[1].set_ylim(-3, 3)
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(True, alpha=0.3)

# Panel C: Random mode reconstruction
axes[2].plot(x, original, 'k-', linewidth=0.8, alpha=0.3, label='Original')
axes[2].plot(x, recon_random, 'r-', linewidth=2, label=f'Random ({k} modes)')
axes[2].set_xlabel('Position')
axes[2].set_title(f'(C) Random mode selection ($k={k}$)', fontsize=11)
axes[2].set_ylim(-3, 3)
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_reconstruction_comparison.pdf')
plt.savefig('fig_reconstruction_comparison.png')
print("Saved fig_reconstruction_comparison.pdf")

# Also save to parent directory for inclusion in paper
import shutil
shutil.copy('fig_reconstruction_comparison.pdf', '../fig_reconstruction_comparison.pdf')
print("Copied to parent directory")
