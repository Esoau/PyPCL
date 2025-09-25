import os
import pandas as pd
from datetime import datetime

def save_accuracies_to_csv(all_accuracies, args, project_root):
    """
    Saves the accuracies for all models and epochs to a single CSV file.
    The CSV is formatted with models as rows and epochs as columns.

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
        f"accuracies_type_{args.type}_value_{args.value}_noise_{args.noise}_eta_{args.eta}_"
        f"batch_size_{args.batch_size}_epochs_{args.epochs}_lr_{args.lr}_"
        f"weight_decay_{args.weight_decay}_momentum_{args.momentum}"
    )
    
    filename = f"{base_filename}_{timestamp}.csv"
    save_path = os.path.join(results_dir, filename)

    # Prepare data for DataFrame
    # Use a copy to avoid modifying the original dictionary
    filtered_accuracies = {k: v for k, v in all_accuracies.items() if v}
    
    if not filtered_accuracies:
        print("No accuracies to save.")
        return

    df = pd.DataFrame.from_dict(filtered_accuracies, orient='index')
    
    # Set column names to be 'Epoch 1', 'Epoch 2', etc.
    df.columns = [f'Epoch {i+1}' for i in range(df.shape[1])]
    
    # Add a 'Model' column from the index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Accuracy results saved to {save_path}")
