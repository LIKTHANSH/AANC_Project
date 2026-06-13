"""
Main pipeline execution script for the Accent-Aware Neural Codec (AANC).
Runs the end-to-end process: data extraction, training, evaluation, and visualization.
"""
import os
import sys
import time

# Add root directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import MODELS_DIR, RESULTS_DIR
from src.dataset import get_data_loaders
from src.train import train_model
from src.evaluate import evaluate_model, create_comparison_table
from src.visualize import generate_all_visualizations

def main():
    print("=" * 80)
    print("STARTING ACCENT-AWARE NEURAL CODEC (AANC) PIPELINE")
    print("=" * 80)
    
    start_time = time.time()

    # 1. Train the model
    # Note: dataset setup/extraction happens inside get_data_loaders called by train_model
    model, history, dataset = train_model()

    # 2. Setup Data Loaders for Evaluation (including test set)
    _, _, test_loader, _ = get_data_loaders()

    # 3. Evaluate the model
    print("\n[Pipeline] Evaluating model...")
    results = evaluate_model(model, test_loader, dataset, save_results=True)

    # 4. Generate comparison table with traditional codecs
    create_comparison_table(results)

    # 5. Generate all visualizations
    print("\n[Pipeline] Generating visualizations and plots...")
    generate_all_visualizations(model, test_loader, dataset, history, results)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed_time/60:.1f} minutes!")
    print(f"Model saved to: {os.path.join(MODELS_DIR, 'aanc_best.pth')}")
    print(f"Plots and results saved to: {RESULTS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
