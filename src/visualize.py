"""
Visualization module for the AANC project.
Generates training curves, spectrograms, waveforms, feature maps,
and comparison plots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import librosa
import librosa.display
from src.config import (
    RESULTS_DIR, SAMPLE_RATE, N_FFT, HOP_LENGTH,
    N_MFCC, N_MELS, MAX_FRAMES, FEATURE_DIM, DEVICE, USE_AMP
)
from torch.amp import autocast

# Style settings
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})


def plot_training_history(history, save_path=None):
    """Plot training and validation loss curves."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "training_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', color='#2196F3', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='#F44336', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Learning rate
    axes[1].plot(history['learning_rate'], color='#4CAF50', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')

    # Epoch time
    axes[2].bar(range(len(history['epoch_time'])), history['epoch_time'],
                color='#FF9800', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Epoch Duration')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Training curves saved to {save_path}")


def plot_feature_comparison(model, test_loader, dataset, n_samples=4, save_path=None):
    """Plot original vs reconstructed feature matrices."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "feature_comparison.png")

    model.eval()
    norm_mean = dataset.mean.to(DEVICE)
    norm_std = dataset.std.to(DEVICE)

    # Get a batch of test data
    features, labels = next(iter(test_loader))
    features = features.to(DEVICE)

    with torch.no_grad():
        if USE_AMP and DEVICE.type == 'cuda':
            with autocast('cuda'):
                recon, latent = model(features)
        else:
            recon, latent = model(features)

    # Denormalize
    features_denorm = (features * norm_std + norm_mean).cpu().numpy()
    recon_denorm = (recon * norm_std + norm_mean).cpu().numpy()

    fig, axes = plt.subplots(n_samples, 3, figsize=(18, 4 * n_samples))

    for i in range(min(n_samples, features.size(0))):
        orig = features_denorm[i, 0]   # (frames, feat_dim)
        rec = recon_denorm[i, 0]
        diff = np.abs(orig - rec)

        # Original
        im1 = axes[i, 0].imshow(orig.T, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Sample {i+1} - Original Features')
        axes[i, 0].set_ylabel('Feature Index')
        plt.colorbar(im1, ax=axes[i, 0])

        # Reconstructed
        im2 = axes[i, 1].imshow(rec.T, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 1].set_title(f'Sample {i+1} - Reconstructed Features')
        plt.colorbar(im2, ax=axes[i, 1])

        # Difference
        im3 = axes[i, 2].imshow(diff.T, aspect='auto', origin='lower', cmap='hot')
        axes[i, 2].set_title(f'Sample {i+1} - Absolute Error')
        plt.colorbar(im3, ax=axes[i, 2])

    for ax in axes[-1]:
        ax.set_xlabel('Frame Index')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Feature comparison saved to {save_path}")


def plot_mfcc_comparison(model, test_loader, dataset, n_samples=3, save_path=None):
    """Plot original vs reconstructed MFCCs specifically."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "mfcc_comparison.png")

    model.eval()
    norm_mean = dataset.mean.to(DEVICE)
    norm_std = dataset.std.to(DEVICE)

    features, labels = next(iter(test_loader))
    features = features.to(DEVICE)

    with torch.no_grad():
        if USE_AMP and DEVICE.type == 'cuda':
            with autocast('cuda'):
                recon, latent = model(features)
        else:
            recon, latent = model(features)

    features_denorm = (features * norm_std + norm_mean).cpu().numpy()
    recon_denorm = (recon * norm_std + norm_mean).cpu().numpy()

    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))

    for i in range(min(n_samples, features.size(0))):
        orig_mfcc = features_denorm[i, 0, :, :N_MFCC].T   # (n_mfcc, frames)
        recon_mfcc = recon_denorm[i, 0, :, :N_MFCC].T

        # Original MFCCs
        librosa.display.specshow(orig_mfcc, x_axis='frames', ax=axes[i, 0],
                                 hop_length=HOP_LENGTH, sr=SAMPLE_RATE)
        axes[i, 0].set_title(f'Sample {i+1} - Original MFCCs')
        axes[i, 0].set_ylabel('MFCC Coefficient')

        # Reconstructed MFCCs
        librosa.display.specshow(recon_mfcc, x_axis='frames', ax=axes[i, 1],
                                 hop_length=HOP_LENGTH, sr=SAMPLE_RATE)
        axes[i, 1].set_title(f'Sample {i+1} - Reconstructed MFCCs')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] MFCC comparison saved to {save_path}")


def plot_pitch_energy_comparison(model, test_loader, dataset, n_samples=3, save_path=None):
    """Plot original vs reconstructed pitch and energy contours."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "pitch_energy_comparison.png")

    model.eval()
    norm_mean = dataset.mean.to(DEVICE)
    norm_std = dataset.std.to(DEVICE)

    features, labels = next(iter(test_loader))
    features = features.to(DEVICE)

    with torch.no_grad():
        if USE_AMP and DEVICE.type == 'cuda':
            with autocast('cuda'):
                recon, latent = model(features)
        else:
            recon, latent = model(features)

    features_denorm = (features * norm_std + norm_mean).cpu().numpy()
    recon_denorm = (recon * norm_std + norm_mean).cpu().numpy()

    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))

    for i in range(min(n_samples, features.size(0))):
        # Pitch is at index -2, Energy at index -1
        orig_pitch = features_denorm[i, 0, :, -2]
        recon_pitch = recon_denorm[i, 0, :, -2]
        orig_energy = features_denorm[i, 0, :, -1]
        recon_energy = recon_denorm[i, 0, :, -1]

        # Pitch
        axes[i, 0].plot(orig_pitch, label='Original', color='#2196F3', linewidth=1.5, alpha=0.8)
        axes[i, 0].plot(recon_pitch, label='Reconstructed', color='#F44336',
                       linewidth=1.5, alpha=0.8, linestyle='--')
        axes[i, 0].set_title(f'Sample {i+1} - Pitch (F0) Contour')
        axes[i, 0].set_ylabel('Pitch (Hz)')
        axes[i, 0].legend()

        # Energy
        axes[i, 1].plot(orig_energy, label='Original', color='#4CAF50', linewidth=1.5, alpha=0.8)
        axes[i, 1].plot(recon_energy, label='Reconstructed', color='#FF9800',
                       linewidth=1.5, alpha=0.8, linestyle='--')
        axes[i, 1].set_title(f'Sample {i+1} - Energy Contour')
        axes[i, 1].set_ylabel('Energy')
        axes[i, 1].legend()

    for ax in axes[-1]:
        ax.set_xlabel('Frame Index')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Pitch/Energy comparison saved to {save_path}")


def plot_latent_space(model, test_loader, dataset, save_path=None):
    """Visualize the latent space using t-SNE, colored by accent."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "latent_space_tsne.png")

    from sklearn.manifold import TSNE

    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            if USE_AMP and DEVICE.type == 'cuda':
                with autocast('cuda'):
                    _, latent = model(features)
            else:
                _, latent = model(features)
            all_latents.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())

    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # t-SNE reduction
    print("[Viz] Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)-1))
    latent_2d = tsne.fit_transform(latents)

    # Get accent names from config
    from src.config import ACCENT_GROUPS
    idx_to_accent = {}
    for i, accent in enumerate(ACCENT_GROUPS.keys()):
        idx_to_accent[i] = accent

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(idx_to_accent)))

    for label_idx in sorted(set(labels)):
        mask = labels == label_idx
        accent_name = idx_to_accent.get(label_idx, f"Unknown_{label_idx}")
        ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                  c=[colors[label_idx % len(colors)]], label=accent_name,
                  alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    ax.set_title('t-SNE Visualization of Latent Space (colored by accent)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Latent space t-SNE saved to {save_path}")


def plot_comparison_bar_chart(aanc_results, save_path=None):
    """Plot bar chart comparing AANC with traditional codecs."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "codec_comparison.png")

    from src.evaluate import get_traditional_codec_metrics
    codec_results = get_traditional_codec_metrics()

    models = list(codec_results.keys()) + ['Proposed AANC']
    metrics_to_plot = {
        'MSE ↓': ([codec_results[m]['MSE'] for m in codec_results] +
                   [aanc_results['MSE']]),
        'PESQ ↑': ([codec_results[m]['PESQ'] for m in codec_results] +
                    [aanc_results.get('PESQ_approx', 0)]),
        'STOI ↑': ([codec_results[m]['STOI'] for m in codec_results] +
                    [aanc_results.get('STOI', 0)]),
        'MCD ↓': ([codec_results[m]['MCD'] for m in codec_results] +
                   [aanc_results['MCD']]),
        'Comp. Ratio ↑': ([codec_results[m]['Compression_Ratio'] for m in codec_results] +
                          [aanc_results['Compression_Ratio']]),
    }

    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    model_colors = ['#90CAF9', '#A5D6A7', '#CE93D8', '#FFD54F']

    for idx, (metric_name, values) in enumerate(metrics_to_plot.items()):
        bars = axes[idx].bar(range(len(models)), values, color=model_colors, edgecolor='white')
        axes[idx].set_title(metric_name, fontweight='bold')
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels([m.replace('_', '\n') for m in models],
                                   fontsize=8, rotation=0)

        # Highlight AANC bar
        bars[-1].set_color('#FF7043')
        bars[-1].set_edgecolor('#E64A19')
        bars[-1].set_linewidth(2)

        # Add value labels
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{val:.3f}' if val < 1 else f'{val:.2f}',
                          ha='center', va='bottom', fontsize=8)

    plt.suptitle('AANC vs Traditional Codecs - Performance Comparison',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Comparison chart saved to {save_path}")


def plot_waveform_comparison(model, test_loader, dataset, n_samples=2, save_path=None):
    """Plot original vs reconstructed waveforms (from MFCC inversion)."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "waveform_comparison.png")

    model.eval()
    norm_mean = dataset.mean.to(DEVICE)
    norm_std = dataset.std.to(DEVICE)

    features, labels = next(iter(test_loader))
    features = features.to(DEVICE)

    with torch.no_grad():
        if USE_AMP and DEVICE.type == 'cuda':
            with autocast('cuda'):
                recon, latent = model(features)
        else:
            recon, latent = model(features)

    features_denorm = (features * norm_std + norm_mean).cpu().numpy()
    recon_denorm = (recon * norm_std + norm_mean).cpu().numpy()

    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))

    for i in range(min(n_samples, features.size(0))):
        orig_mfcc = features_denorm[i, 0, :, :N_MFCC].T
        recon_mfcc = recon_denorm[i, 0, :, :N_MFCC].T

        try:
            orig_audio = librosa.feature.inverse.mfcc_to_audio(
                orig_mfcc, n_mels=80, sr=SAMPLE_RATE,
                n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            recon_audio = librosa.feature.inverse.mfcc_to_audio(
                recon_mfcc, n_mels=80, sr=SAMPLE_RATE,
                n_fft=N_FFT, hop_length=HOP_LENGTH
            )

            # Normalize
            if np.max(np.abs(orig_audio)) > 0:
                orig_audio /= np.max(np.abs(orig_audio))
            if np.max(np.abs(recon_audio)) > 0:
                recon_audio /= np.max(np.abs(recon_audio))

            t_orig = np.arange(len(orig_audio)) / SAMPLE_RATE
            t_recon = np.arange(len(recon_audio)) / SAMPLE_RATE

            axes[i, 0].plot(t_orig, orig_audio, color='#2196F3', linewidth=0.5)
            axes[i, 0].set_title(f'Sample {i+1} - Original Waveform')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].set_ylim(-1.1, 1.1)

            axes[i, 1].plot(t_recon, recon_audio, color='#F44336', linewidth=0.5)
            axes[i, 1].set_title(f'Sample {i+1} - Reconstructed Waveform')
            axes[i, 1].set_ylim(-1.1, 1.1)

        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f'Error: {e}', transform=axes[i, 0].transAxes,
                          ha='center')
            axes[i, 1].text(0.5, 0.5, f'Error: {e}', transform=axes[i, 1].transAxes,
                          ha='center')

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Waveform comparison saved to {save_path}")


def plot_spectrogram_comparison(model, test_loader, dataset, n_samples=2, save_path=None):
    """Plot mel spectrogram comparison (original vs reconstructed)."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "spectrogram_comparison.png")

    model.eval()
    norm_mean = dataset.mean.to(DEVICE)
    norm_std = dataset.std.to(DEVICE)

    features, labels = next(iter(test_loader))
    features = features.to(DEVICE)

    with torch.no_grad():
        if USE_AMP and DEVICE.type == 'cuda':
            with autocast('cuda'):
                recon, latent = model(features)
        else:
            recon, latent = model(features)

    features_denorm = (features * norm_std + norm_mean).cpu().numpy()
    recon_denorm = (recon * norm_std + norm_mean).cpu().numpy()

    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 5 * n_samples))

    for i in range(min(n_samples, features.size(0))):
        orig_mfcc = features_denorm[i, 0, :, :N_MFCC].T
        recon_mfcc = recon_denorm[i, 0, :, :N_MFCC].T

        try:
            orig_audio = librosa.feature.inverse.mfcc_to_audio(
                orig_mfcc, n_mels=80, sr=SAMPLE_RATE,
                n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            recon_audio = librosa.feature.inverse.mfcc_to_audio(
                recon_mfcc, n_mels=80, sr=SAMPLE_RATE,
                n_fft=N_FFT, hop_length=HOP_LENGTH
            )

            # Compute mel spectrograms
            orig_mel = librosa.feature.melspectrogram(y=orig_audio, sr=SAMPLE_RATE,
                                                      n_fft=N_FFT, hop_length=HOP_LENGTH)
            recon_mel = librosa.feature.melspectrogram(y=recon_audio, sr=SAMPLE_RATE,
                                                       n_fft=N_FFT, hop_length=HOP_LENGTH)

            orig_mel_db = librosa.power_to_db(orig_mel, ref=np.max)
            recon_mel_db = librosa.power_to_db(recon_mel, ref=np.max)

            librosa.display.specshow(orig_mel_db, x_axis='time', y_axis='mel',
                                    sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                    ax=axes[i, 0])
            axes[i, 0].set_title(f'Sample {i+1} - Original Mel Spectrogram')

            librosa.display.specshow(recon_mel_db, x_axis='time', y_axis='mel',
                                    sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                    ax=axes[i, 1])
            axes[i, 1].set_title(f'Sample {i+1} - Reconstructed Mel Spectrogram')

        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f'Error: {e}', transform=axes[i, 0].transAxes,
                          ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Spectrogram comparison saved to {save_path}")


def generate_all_visualizations(model, test_loader, dataset, history, aanc_results):
    """Generate all visualizations."""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_training_history(history)
    plot_feature_comparison(model, test_loader, dataset)
    plot_mfcc_comparison(model, test_loader, dataset)
    plot_pitch_energy_comparison(model, test_loader, dataset)
    plot_waveform_comparison(model, test_loader, dataset)
    plot_spectrogram_comparison(model, test_loader, dataset)
    plot_latent_space(model, test_loader, dataset)
    plot_comparison_bar_chart(aanc_results)

    print("\n[Viz] All visualizations generated in:", RESULTS_DIR)
