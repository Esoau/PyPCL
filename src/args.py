import argparse

def parse_arguments(data_config, train_config):
    """
    Parses command-line arguments for the experiment.
    
    Args:
        data_config (dict): Dictionary containing data generation configurations.
        train_config (dict): Dictionary containing training configurations.
        
    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Generate weak labels and train models.')
    
    # Label generation arguments
    parser.add_argument('--type', choices=['constant', 'variable'], required=True, help='Type of label generation.')
    parser.add_argument('--value', type=float, required=True, help='Value for k (if type=constant) or q (if type=variable).')
    
    # Noise arguments
    parser.add_argument('--noise', choices=['noisy', 'clean'], default='clean', help='Type of noise to add.')
    parser.add_argument('--eta', type=float, default=data_config.get('eta', 0.0), help='Noise level eta. Only used if noise is noisy.')
    
    # Training arguments from config
    parser.add_argument('--batch_size', type=int, default=train_config['batch_size'], help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=train_config['epochs'], help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=train_config['lr'], help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help='Weight decay for optimizer.')
    parser.add_argument('--momentum', type=float, default=train_config['momentum'], help='Momentum for optimizer.')
    
    args = parser.parse_args()
    return args