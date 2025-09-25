import os
import pandas as pd
from datetime import datetime

def save_accuracies_to_csv(all_accuracies, args, project_root):
    """
    Saves the final best accuracies for all models to a single CSV file.

    Args:
        all_accuracies (dict): A dictionary containing accuracy lists for each model.
        args (Namespace): Arguments containing hyperparameters.
        project_root (str): The root directory of the project.
    """
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a base filename from hyperparameters
    base_filename = (
        f"type_{args.type}_value_{args.value}_noise_{args.noise}_eta_{args.eta}_"
        f"epochs_{args.epochs}_lr_{args.lr}"
    )
    
    filename = f"accuracies_{base_filename}_{timestamp}.csv"
    save_path = os.path.join(results_dir, filename)

    # Prepare data for DataFrame
    data = {
        'Model': list(all_accuracies.keys()),
        'Best Accuracy (%)': [max(accs) if accs else 0 for accs in all_accuracies.values()]
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Accuracy results saved to {save_path}")
