"""
Configuration file for the Accent-Aware Neural Codec (AANC) project.
Contains all hyperparameters, paths, and settings.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import torch

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VCTK_ZIP = os.path.join(DATA_DIR, "DS_10283_3443", "VCTK-Corpus-0.92.zip")
VCTK_EXTRACTED = os.path.join(DATA_DIR, "VCTK-Corpus-0.92")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

# Create directories
for d in [RESULTS_DIR, MODELS_DIR, FEATURES_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Audio Parameters
# =============================================================================
SAMPLE_RATE = 16000       # Target sample rate (Hz)
FRAME_SIZE_MS = 25        # Frame size in milliseconds
HOP_SIZE_MS = 10          # Hop size in milliseconds
N_FFT = int(SAMPLE_RATE * FRAME_SIZE_MS / 1000)   # 400 samples
HOP_LENGTH = int(SAMPLE_RATE * HOP_SIZE_MS / 1000) # 160 samples
MAX_AUDIO_LENGTH_SEC = 4  # Max audio length in seconds for uniform input

# =============================================================================
# Feature Extraction Parameters
# =============================================================================
N_MFCC = 13              # Number of MFCC coefficients
USE_DELTA = True          # Use delta and delta-delta MFCCs
N_MELS = 80              # Number of mel bands for spectrogram
FEATURE_DIM = N_MFCC * 3 + 2 if USE_DELTA else N_MFCC + 2  # 41 = 39 MFCC + pitch + energy

# Fixed number of frames for uniform input size
MAX_FRAMES = int(MAX_AUDIO_LENGTH_SEC * SAMPLE_RATE / HOP_LENGTH)  # 400 frames for 4s

# =============================================================================
# Model Architecture
# =============================================================================
LATENT_DIM = 512          # Latent space dimensionality (32x compression)
ENCODER_CHANNELS = [1, 32, 64, 128]
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
NUM_WORKERS = 0
USE_AMP = True            # Automatic Mixed Precision for GPU

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[Config] GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# Speaker / Accent Configuration
# =============================================================================
# Accent groups from VCTK corpus
ACCENT_GROUPS = {
    "English": ["p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232",
                "p233", "p236", "p239", "p240", "p243", "p244", "p250", "p254",
                "p256", "p257", "p258", "p259", "p267", "p268", "p269", "p270",
                "p273", "p274", "p276", "p277", "p278", "p279", "p282", "p286", "p287"],
    "Scottish": ["p234", "p237", "p241", "p246", "p247", "p249", "p252", "p255",
                 "p260", "p262", "p263", "p264", "p265", "p271", "p272", "p275",
                 "p281", "p284", "p285"],
    "American": ["p294", "p297", "p299", "p300", "p301", "p305", "p306", "p308",
                 "p310", "p311", "p318", "p329", "p330", "p333", "p334", "p339",
                 "p341", "p345", "p360", "p361", "p362"],
    "Irish": ["p245", "p266", "p283", "p288", "p295", "p298", "p313", "p340", "p364"],
    "Indian": ["p248", "p251", "p376"],
    "NorthernIrish": ["p238", "p261", "p292", "p293", "p304", "p351"],
    "Canadian": ["p302", "p303", "p307", "p312", "p316", "p317", "p343", "p363"],
    "SouthAfrican": ["p314", "p323", "p336", "p347"],
    "Welsh": ["p253"],
    "Australian": ["p326", "p374"],
    "NewZealand": ["p335"],
}

# Select a diverse subset of speakers for training (balanced across accents)
SELECTED_SPEAKERS = []
MAX_SPEAKERS_PER_ACCENT = 5  # Take up to 5 speakers per accent group
for accent, speakers in ACCENT_GROUPS.items():
    SELECTED_SPEAKERS.extend(speakers[:MAX_SPEAKERS_PER_ACCENT])

MAX_UTTERANCES_PER_SPEAKER = 50  # Limit utterances per speaker to keep dataset manageable
