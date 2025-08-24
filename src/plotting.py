import matplotlib.pyplot as plt
import os

def save_accuracy_plot(accuracies_dict, epochs_range, args, project_root):
    """Saves the accuracy plot to a file."""
    plt.figure(figsize=(12, 8))

    for model_name, accuracies in accuracies_dict.items():
        if accuracies:
            plt.plot(epochs_range, accuracies, '-', label=f'{model_name} Test Accuracy')

    # Create title with all arguments
    args_str = ', '.join(f'{k}={v}' for k, v in vars(args).items())
    plt.title(f'Test Accuracy vs. Epochs\n({args_str})', fontsize=10)
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for long title

    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a filename from the arguments
    args_filename = '_'.join(f'{k}_{v}' for k, v in vars(args).items()).replace('.', '_')
    filename = f'accuracy_plot_{args_filename}.png'
    
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory
    print(f"Plot updated and saved to {save_path}")