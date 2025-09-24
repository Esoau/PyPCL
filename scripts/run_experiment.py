import os
import sys
import torch
import yaml
import matplotlib.pyplot as plt
import gc

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
from src.data_setup import prepare_datasets, get_dataloaders
from src.training_pipelines import run_proden_training, run_mcl_training, run_pico_training, run_solar_training
from src.plotting import save_accuracy_plot
from src.args import parse_arguments

def main():
    # Configuration and Argument Parsing
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    args = parse_arguments(config['data_generation'], config['training'])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Data Preparation
    pl_dataset_raw, cl_dataset_raw, original_targets = prepare_datasets(args, config['data_generation'], config['training'])
    loaders, solar_train_dataset = get_dataloaders(args, config['data_generation'], pl_dataset_raw, cl_dataset_raw, original_targets)

    # Accuracy tracking
    all_accuracies = {
        'PRODEN': [], 'MCL-LOG': [], 'MCL-MAE': [], 'MCL-EXP': [], 'PiCO': [], 'SoLar': []
    }
    epochs_range = range(1, args.epochs + 1)

    # --- Training Pipelines ---

    # PRODEN
    all_accuracies['PRODEN'] = run_proden_training(args, loaders, config['training'], DEVICE)
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)

    # MCL-LOG
    all_accuracies['MCL-LOG'] = run_mcl_training(args, loaders, config['training'], DEVICE, 'log')
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)

    # MCL-MAE
    all_accuracies['MCL-MAE'] = run_mcl_training(args, loaders, config['training'], DEVICE, 'mae')
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)

    # MCL-EXP
    all_accuracies['MCL-EXP'] = run_mcl_training(args, loaders, config['training'], DEVICE, 'exp')
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)

    # PiCO
    all_accuracies['PiCO'] = run_pico_training(args, loaders, config['training'], config['pico'], pl_dataset_raw, original_targets, DEVICE)
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)

    # SoLar
    all_accuracies['SoLar'] = run_solar_training(args, loaders, config['training'], config['solar'], solar_train_dataset, DEVICE)
    save_accuracy_plot(all_accuracies, epochs_range, args, project_root)
    
    # Final Cleanup
    del loaders, pl_dataset_raw, cl_dataset_raw, original_targets, solar_train_dataset
    gc.collect()

    # Final Results
    print("\n--- Final Results ---")
    for name, accs in all_accuracies.items():
        if accs:
            print(f"Best Accuracy ({name}): {max(accs):.2f}%")

    plt.show()

if __name__ == "__main__":
    main()