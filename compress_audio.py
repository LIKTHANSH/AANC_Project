"""
Utility script to compress and decompress any raw audio file using the trained AANC model.
"""
import os
import sys
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa

# Add root directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DEVICE, FEATURE_DIM, LATENT_DIM, MAX_FRAMES, SAMPLE_RATE, N_MFCC
from src.features import extract_features
from src.model import ConvAutoencoder

def load_codec_model(model_path):
    """Load the trained model and normalization parameters from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")
        
    print(f"[Codec] Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = ConvAutoencoder()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    mean = checkpoint['norm_mean'].to(DEVICE)
    std = checkpoint['norm_std'].to(DEVICE)
    
    print("[Codec] Model loaded successfully.")
    return model, mean, std

def compress(audio_path, model, mean, std, output_latent_path):
    """Compress raw audio into the latent space and save to file."""
    print(f"\n[Compress] Loading audio from: {audio_path}")
    
    # Load and preprocess audio
    audio, orig_sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        
    if orig_sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
        
    # Max normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
        
    # Trim or pad to fixed length (matching train setup)
    max_samples = int(4 * SAMPLE_RATE)  # 4 seconds
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
        
    # Extract features
    print("[Compress] Extracting acoustic and prosodic features (MFCC, pitch, energy)...")
    feats = extract_features(audio, sr=SAMPLE_RATE)
    
    # Standardize length to MAX_FRAMES
    if feats.shape[0] > MAX_FRAMES:
        feats = feats[:MAX_FRAMES]
    elif feats.shape[0] < MAX_FRAMES:
        feats = np.pad(feats, ((0, MAX_FRAMES - feats.shape[0]), (0, 0)), mode='constant')
        
    # Prepare tensor
    feats_tensor = torch.FloatTensor(feats).unsqueeze(0).unsqueeze(1).to(DEVICE)  # [1, 1, MAX_FRAMES, FEATURE_DIM]
    
    # Normalize features
    feats_norm = (feats_tensor - mean) / std
    
    # Encode to latent space
    print("[Compress] Running neural encoder...")
    with torch.no_grad():
        latent = model.encode(feats_norm)  # [1, LATENT_DIM] (handles skip connections internally)
        latent_np = latent.cpu().numpy().astype(np.float32)
        
    # Save latent representation
    np.save(output_latent_path, latent_np)
    
    orig_size_bytes = os.path.getsize(audio_path)
    latent_size_bytes = os.path.getsize(output_latent_path)
    comp_ratio = orig_size_bytes / latent_size_bytes if latent_size_bytes > 0 else 0
    
    print("-" * 50)
    print("COMPRESSION SUMMARY:")
    print(f"  Original Audio File:  {audio_path} ({orig_size_bytes/1024:.2f} KB)")
    print(f"  Compressed Latent:    {output_latent_path} ({latent_size_bytes/1024:.2f} KB)")
    print(f"  Acoustic Comp Ratio:  {comp_ratio:.1f}x (compared to original file size)")
    print(f"  Feature-level Comp:   {(MAX_FRAMES * FEATURE_DIM * 4) / (LATENT_DIM * 4):.1f}x")
    print("-" * 50)
    
    return latent_np

def decompress(latent_path, model, mean, std, output_wav_path):
    """Decompress latent code back to raw audio."""
    print(f"\n[Decompress] Loading compressed latent from: {latent_path}")
    latent_np = np.load(latent_path)
    latent_tensor = torch.FloatTensor(latent_np).to(DEVICE)  # [1, LATENT_DIM]
    
    # Decode to reconstructed features
    print("[Decompress] Running neural decoder...")
    with torch.no_grad():
        recon_norm = model.decode(latent_tensor)  # [1, 1, MAX_FRAMES, FEATURE_DIM]
        
    # Denormalize
    recon_feat_tensor = recon_norm * std + mean
    recon_feats = recon_feat_tensor.squeeze(0).squeeze(0).cpu().numpy()  # [MAX_FRAMES, FEATURE_DIM]
    
    # Extract reconstructed MFCCs
    recon_mfcc = recon_feats[:, :N_MFCC].T  # [N_MFCC, MAX_FRAMES]
    
    # Reconstruct audio via inverse MFCC
    print("[Decompress] Reconstructing waveform from spectral & prosodic contours...")
    recon_audio = librosa.feature.inverse.mfcc_to_audio(
        recon_mfcc, n_mels=80, sr=SAMPLE_RATE,
        n_fft=400, hop_length=160
    )
    
    # Normalize reconstructed audio
    if np.max(np.abs(recon_audio)) > 0:
        recon_audio = recon_audio / np.max(np.abs(recon_audio))
        
    # Save reconstructed wav
    sf.write(output_wav_path, recon_audio, SAMPLE_RATE)
    print(f"[Decompress] Reconstructed audio saved to: {output_wav_path}")
    return recon_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AANC Audio Compression Utility")
    parser.add_argument("--mode", choices=["compress", "decompress", "both"], default="both",
                        help="Mode of operation: compress, decompress, or both")
    parser.add_argument("--input", required=True, help="Path to input audio file (for compress/both) or latent file (for decompress)")
    parser.add_argument("--latent", default="compressed.npy", help="Path to save/load compressed latent representation")
    parser.add_argument("--output", default="reconstructed.wav", help="Path to save reconstructed audio (for decompress/both)")
    parser.add_argument("--model", default="models/aanc_best.pth", help="Path to trained model checkpoint")
    
    args = parser.parse_args()
    
    try:
        model, mean, std = load_codec_model(args.model)
        
        if args.mode in ["compress", "both"]:
            compress(args.input, model, mean, std, args.latent)
            
        if args.mode in ["decompress", "both"]:
            latent_to_use = args.latent if args.mode == "both" else args.input
            decompress(latent_to_use, model, mean, std, args.output)
            
    except Exception as e:
        print(f"[Error] Codec operation failed: {e}")
        sys.exit(1)
