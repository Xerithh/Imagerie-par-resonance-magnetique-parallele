#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from utils import pMRI_simulator, reconstruct, reconstruct_tikhonov, SignalToNoiseRatio


def array_snr_db(reference, estimate):
    reference = np.asarray(reference)
    estimate = np.asarray(estimate)
    error = reference - estimate
    den = np.linalg.norm(error.ravel())
    if den == 0:
        return float("inf")
    return 20 * np.log10(np.linalg.norm(reference.ravel()) / den)


def simulate_for_R(S, ref, R, sigma_values):
    clean = pMRI_simulator(S, ref, 0, R)
    runs = {}
    stats = {}
    Nc = S.shape[2]

    for sigma in sigma_values:
        data = pMRI_simulator(S, ref, sigma, R)
        runs[sigma] = data

        diff = data - clean
        coil_snr = [array_snr_db(clean[:, :, k], data[:, :, k]) for k in range(Nc)]
        coil_std = [float(np.std(diff[:, :, k])) for k in range(Nc)]

        stats[sigma] = {
            "global_snr": array_snr_db(clean, data),
            "coil_snr_min": float(np.min(coil_snr)),
            "coil_snr_mean": float(np.mean(coil_snr)),
            "coil_snr_max": float(np.max(coil_snr)),
            "coil_std_min": float(np.min(coil_std)),
            "coil_std_mean": float(np.mean(coil_std)),
            "coil_std_max": float(np.max(coil_std)),
        }

    return runs, stats


def plot_simulations_by_sigma(runs, sigma_values, n_show, out_name, title):
    n_rows = len(sigma_values)
    n_cols = n_show + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, sigma in enumerate(sigma_values):
        data = runs[sigma]
        rss = np.sqrt(np.sum(np.abs(data) ** 2, axis=2))
        axes[i, 0].imshow(rss, cmap="gray")
        axes[i, 0].set_title(f"sigma={sigma} RSS")
        axes[i, 0].axis("off")

        for c in range(n_show):
            axes[i, c + 1].imshow(np.abs(data[:, :, c]), cmap="gray")
            axes[i, c + 1].set_title(f"Antenne {c + 1}")
            axes[i, c + 1].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_name, dpi=180)
    plt.close(fig)


def plot_q2_compare_sigma14(runs_r2, runs_r4, sigma, n_show, out_name):
    fig, axes = plt.subplots(2, n_show + 1, figsize=(2.6 * (n_show + 1), 5.0))

    for row, (R, runs) in enumerate([(2, runs_r2), (4, runs_r4)]):
        data = runs[sigma]
        rss = np.sqrt(np.sum(np.abs(data) ** 2, axis=2))
        axes[row, 0].imshow(rss, cmap="gray")
        axes[row, 0].set_title(f"R={R}, sigma={sigma}, RSS")
        axes[row, 0].axis("off")

        for c in range(n_show):
            axes[row, c + 1].imshow(np.abs(data[:, :, c]), cmap="gray")
            axes[row, c + 1].set_title(f"R={R}, Antenne {c + 1}")
            axes[row, c + 1].axis("off")

    fig.suptitle("Q2: Comparaison R=2 vs R=4 pour sigma=14")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_name, dpi=180)
    plt.close(fig)


def plot_q3_recon(ref, recon_r2, recon_r4, out_name):
    err_r2 = np.abs(ref - recon_r2)
    err_r4 = np.abs(ref - recon_r4)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].imshow(np.abs(ref), cmap="gray")
    axes[0, 0].set_title("Reference")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.abs(recon_r2), cmap="gray")
    axes[0, 1].set_title("SENSE R=2")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.abs(recon_r4), cmap="gray")
    axes[0, 2].set_title("SENSE R=4")
    axes[0, 2].axis("off")

    axes[1, 0].axis("off")
    axes[1, 1].imshow(err_r2, cmap="hot")
    axes[1, 1].set_title("Erreur R=2")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(err_r4, cmap="hot")
    axes[1, 2].set_title("Erreur R=4")
    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_name, dpi=180)
    plt.close(fig)


def plot_q4_lambda(lambda_values, snr_values, out_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(lambda_values, snr_values, marker="o")
    ax.set_xlabel("lambda")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Q4: SNR en fonction de lambda")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_name, dpi=180)
    plt.close(fig)


def plot_q4_compare(ref, sense_r4, tikh_r4, best_lambda, out_name):
    err_sense = np.abs(ref - sense_r4)
    err_tikh = np.abs(ref - tikh_r4)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].imshow(np.abs(ref), cmap="gray")
    axes[0, 0].set_title("Reference")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.abs(sense_r4), cmap="gray")
    axes[0, 1].set_title("SENSE R=4")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.abs(tikh_r4), cmap="gray")
    axes[0, 2].set_title(f"Tikhonov R=4, lambda={best_lambda:.2e}")
    axes[0, 2].axis("off")

    axes[1, 0].axis("off")
    axes[1, 1].imshow(err_sense, cmap="hot")
    axes[1, 1].set_title("Erreur SENSE")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(err_tikh, cmap="hot")
    axes[1, 2].set_title("Erreur Tikhonov")
    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_name, dpi=180)
    plt.close(fig)


loaded = scipy.io.loadmat("reference.mat")
ref = loaded["im"]
loaded = scipy.io.loadmat("sens.mat")
S = loaded["s"]
Nc = S.shape[2]

np.random.seed(0)
sigma_values = [0, 5, 14, 30]
n_show = min(4, Nc)
runs_r2, stats_r2 = simulate_for_R(S, ref, 2, sigma_values)
runs_r4, stats_r4 = simulate_for_R(S, ref, 4, sigma_values)

print("\nQ1 - R=2")
for sigma in sigma_values:
    s = stats_r2[sigma]
    print(
        f"sigma={sigma} | SNR_global={s['global_snr']:.2f} dB | "
        f"SNR_coils={s['coil_snr_min']:.2f}/{s['coil_snr_mean']:.2f}/{s['coil_snr_max']:.2f} dB | "
        f"STD_bruit={s['coil_std_min']:.2f}/{s['coil_std_mean']:.2f}/{s['coil_std_max']:.2f}"
    )

print("\nQ2 - R=4")
for sigma in sigma_values:
    s = stats_r4[sigma]
    print(
        f"sigma={sigma} | SNR_global={s['global_snr']:.2f} dB | "
        f"SNR_coils={s['coil_snr_min']:.2f}/{s['coil_snr_mean']:.2f}/{s['coil_snr_max']:.2f} dB | "
        f"STD_bruit={s['coil_std_min']:.2f}/{s['coil_std_mean']:.2f}/{s['coil_std_max']:.2f}"
    )

plot_simulations_by_sigma(runs_r2, sigma_values, n_show, "q1_r2_noise.png", "Q1: R=2, simulations pour differents sigma")
plot_simulations_by_sigma(runs_r4, sigma_values, n_show, "q2_r4_noise.png", "Q2: R=4, simulations pour differents sigma")
plot_q2_compare_sigma14(runs_r2, runs_r4, 14, n_show, "q2_r2_vs_r4_sigma14.png")

sigma_recon = 14
psi = (sigma_recon ** 2) * np.eye(Nc)
recon_r2 = reconstruct(runs_r2[sigma_recon], S, psi)
recon_r4 = reconstruct(runs_r4[sigma_recon], S, psi)
snr_r2 = SignalToNoiseRatio(ref, np.abs(recon_r2))
snr_r4 = SignalToNoiseRatio(ref, np.abs(recon_r4))

print("\nQ3 - Reconstruction SENSE")
print(f"SNR_R2={snr_r2:.2f} dB")
print(f"SNR_R4={snr_r4:.2f} dB")

plot_q3_recon(ref, np.abs(recon_r2), np.abs(recon_r4), "q3_sense_reconstruction.png")

lambda_values = np.logspace(-4, 2, 12)
snr_lambda = []
best_lambda = None
best_snr = -np.inf
best_recon = None

for lambd in lambda_values:
    recon_tikh = reconstruct_tikhonov(runs_r4[sigma_recon], S, psi, lambd)
    snr_val = SignalToNoiseRatio(ref, np.abs(recon_tikh))
    snr_lambda.append(snr_val)
    print(f"Q4 lambda={lambd:.2e} -> SNR={snr_val:.2f} dB")
    if snr_val > best_snr:
        best_snr = snr_val
        best_lambda = lambd
        best_recon = recon_tikh

print("\nQ4 - Tikhonov")
print(f"best_lambda={best_lambda:.2e}")
print(f"SNR_Tikhonov={best_snr:.2f} dB")
print(f"SNR_SENSE_R4={snr_r4:.2f} dB")

plot_q4_lambda(lambda_values, snr_lambda, "q4_lambda_curve.png")
plot_q4_compare(ref, np.abs(recon_r4), np.abs(best_recon), best_lambda, "q4_sense_vs_tikhonov.png")
