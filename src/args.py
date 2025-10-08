import argparse

def parse_arguments(data_config, train_config):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate weak labels and train models.')
    
    # Dataset selection
    parser.add_argument('--dataset', choices=['cifar10', 'cifar20', 'clcifar10', 'clcifar20'], default='cifar10', help='Dataset to use.')

    # Label generation arguments
    parser.add_argument('--type', choices=['constant', 'variable'], help='Label generation type for CIFAR-10/20.')
    parser.add_argument('--value', type=float, help='Value for k (constant) or q (variable).')
    
    # Noise arguments
    parser.add_argument('--noise', choices=['noisy', 'clean'], default='clean', help='Noise type for CIFAR-10/20.')
    parser.add_argument('--eta', type=float, default=data_config.get('eta', 0.0), help='Noise level eta for noisy labels.')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=train_config['batch_size'], help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=train_config['epochs'], help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=train_config['lr'], help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help='Weight decay for optimizer.')
    parser.add_argument('--momentum', type=float, default=train_config['momentum'], help='Momentum for optimizer.')
    
    args = parser.parse_args()

    # Validate arguments for CLCIFAR datasets
    if args.dataset in ['clcifar10', 'clcifar20']:
        if args.type is not None or args.noise != 'clean':
            parser.error(f"--type and --noise are not supported for {args.dataset}.")
    # Validate arguments for CIFAR datasets
    elif args.dataset in ['cifar10', 'cifar20']:
        if args.type is None or args.value is None:
            parser.error(f"--type and --value are required for {args.dataset}.")

    return args