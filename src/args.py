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
    
    # Dataset selection
    parser.add_argument('--dataset', choices=['cifar10', 'clcifar10'], default='cifar10', help='Dataset to use.')

    # Label generation arguments
    parser.add_argument('--type', choices=['constant', 'variable'], help='Type of label generation for CIFAR-10.')
    parser.add_argument('--value', type=float, help='Value for k (if type=constant) or q (if type=variable).')
    
    # Noise arguments
    parser.add_argument('--noise', choices=['noisy', 'clean'], default='clean', help='Type of noise to add for CIFAR-10.')
    parser.add_argument('--eta', type=float, default=data_config.get('eta', 0.0), help='Noise level eta. Only used if noise is noisy.')
    
    # Training arguments from config
    parser.add_argument('--batch_size', type=int, default=train_config['batch_size'], help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=train_config['epochs'], help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=train_config['lr'], help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help='Weight decay for optimizer.')
    parser.add_argument('--momentum', type=float, default=train_config['momentum'], help='Momentum for optimizer.')
    
    args = parser.parse_args()

    if args.dataset == 'clcifar10':
        if args.type is not None or args.noise != 'clean':
            parser.error("--type and --noise arguments are not supported for the clcifar10 dataset.")
    elif args.dataset == 'cifar10':
        if args.type is None or args.value is None:
            parser.error("--type and --value are required for the cifar10 dataset.")

    return args