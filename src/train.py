"""
Training module for the AANC project.
Implements GPU-accelerated training with mixed precision, combined loss functions,
cosine annealing with warm restarts, early stopping, and comprehensive logging.
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler
from src.config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    USE_AMP, MODELS_DIR, RESULTS_DIR, N_MFCC, FEATURE_DIM
)
from src.model import ConvAutoencoder
from src.dataset import get_data_loaders


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss for perceptual quality."""

    def forward(self, recon, target):
        return torch.norm(target - recon, p='fro') / (torch.norm(target, p='fro') + 1e-8)


class CombinedLoss(nn.Module):
    """
    Combined loss function for better perceptual reconstruction:
      - MSE loss (overall reconstruction)
      - Spectral convergence loss (perceptual quality)
      - MFCC emphasis loss (accent-critical features get extra weight)
    """

    def __init__(self, mse_weight=1.0, spectral_weight=0.5, mfcc_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.mfcc_weight = mfcc_weight
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = SpectralConvergenceLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, recon, target):
        # Overall MSE
        loss_mse = self.mse_loss(recon, target)

        # Spectral convergence
        loss_spectral = self.spectral_loss(recon, target)

        # Extra weight on MFCC features (first N_MFCC columns are critical for accent)
        mfcc_recon = recon[:, :, :, :N_MFCC]
        mfcc_target = target[:, :, :, :N_MFCC]
        loss_mfcc = self.l1_loss(mfcc_recon, mfcc_target)

        total = (self.mse_weight * loss_mse +
                 self.spectral_weight * loss_spectral +
                 self.mfcc_weight * loss_mfcc)

        return total, {
            'mse': loss_mse.item(),
            'spectral': loss_spectral.item(),
            'mfcc_l1': loss_mfcc.item(),
            'total': total.item()
        }


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=30, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_model():
    """Train the U-Net Convolutional Autoencoder model."""
    print("=" * 70)
    print("ACCENT-AWARE NEURAL CODEC (AANC) - TRAINING")
    print("=" * 70)

    # Load data
    print("\n[Training] Loading data...")
    train_loader, val_loader, test_loader, dataset = get_data_loaders()

    # Initialize model
    print("\n[Training] Initializing model...")
    model = ConvAutoencoder()
    model = model.to(DEVICE)

    # Combined loss function
    criterion = CombinedLoss(mse_weight=1.0, spectral_weight=0.5, mfcc_weight=0.3)

    # Optimizer with slightly lower LR for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda') if USE_AMP and DEVICE.type == 'cuda' else None

    # Early stopping with more patience
    early_stopping = EarlyStopping(patience=30)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'learning_rate': [], 'epoch_time': []
    }

    best_val_loss = float('inf')
    best_model_path = os.path.join(MODELS_DIR, "aanc_best.pth")

    print(f"\n[Training] Starting training for {EPOCHS} epochs...")
    print(f"[Training] Device: {DEVICE}")
    print(f"[Training] AMP enabled: {USE_AMP and DEVICE.type == 'cuda'}")
    print(f"[Training] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"[Training] Loss: MSE + Spectral Convergence + MFCC L1")
    print("-" * 70)

    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ---- Training Phase ----
        model.train()
        train_losses = []

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda'):
                    recon, latent = model(features)
                    loss, loss_dict = criterion(recon, features)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, latent = model(features)
                loss, loss_dict = criterion(recon, features)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(loss_dict['total'])

        avg_train_loss = np.mean(train_losses)

        # ---- Validation Phase ----
        model.eval()
        val_losses = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE, non_blocking=True)

                if USE_AMP and DEVICE.type == 'cuda':
                    with autocast('cuda'):
                        recon, latent = model(features)
                        loss, loss_dict = criterion(recon, features)
                else:
                    recon, latent = model(features)
                    loss, loss_dict = criterion(recon, features)

                val_losses.append(loss_dict['total'])

        avg_val_loss = np.mean(val_losses)

        # Update scheduler
        scheduler.step()

        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history,
                'norm_mean': dataset.mean,
                'norm_std': dataset.std,
            }, best_model_path)
            save_marker = " * SAVED"
        else:
            save_marker = ""

        # Print progress
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s{save_marker}")

        # Check early stopping
        if early_stopping(avg_val_loss):
            print(f"\n[Training] Early stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - total_start
    print("-" * 70)
    print(f"[Training] Training complete in {total_time/60:.1f} minutes")
    print(f"[Training] Best validation loss: {best_val_loss:.6f}")
    print(f"[Training] Best model saved to: {best_model_path}")

    # Load best model for evaluation
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[Training] Loaded best model checkpoint for evaluation")

    # Save final model
    final_model_path = os.path.join(MODELS_DIR, "aanc_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'norm_mean': dataset.mean,
        'norm_std': dataset.std,
    }, final_model_path)

    # Save training history
    history_path = os.path.join(RESULTS_DIR, "training_history.npz")
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    print(f"[Training] Training history saved to: {history_path}")

    return model, history, dataset


def load_trained_model(model_path=None):
    """Load a trained model from checkpoint."""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "aanc_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = ConvAutoencoder()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"[Model] Loaded model from {model_path}")
    print(f"[Model] Trained for {checkpoint['epoch']+1} epochs")
    print(f"[Model] Best val loss: {checkpoint.get('val_loss', 'N/A')}")

    return model, checkpoint


if __name__ == "__main__":
    train_model()
