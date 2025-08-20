import matplotlib.pyplot as plt
import os

def save_accuracy_plot(accuracies_dict, epochs_range, args, project_root):
    """Saves the accuracy plot to a file."""
    plt.figure(figsize=(12, 7))

    for model_name, accuracies in accuracies_dict.items():
        if accuracies:
            plt.plot(epochs_range, accuracies, '-', label=f'{model_name} Test Accuracy')

    if args.type == 'constant':
        plt.title(f'Test Accuracy vs. Epochs (Constant k={int(args.value)})')
    else:
        plt.title(f'Test Accuracy vs. Epochs (Variable q={args.value})')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if args.type == 'constant':
        filename = f'accuracy_plot_k_{int(args.value)}.png'
    else:
        filename = f'accuracy_plot_q_{args.value}.png'
    
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory
    print(f"Plot updated and saved to {save_path}")