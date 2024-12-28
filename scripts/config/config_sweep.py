import wandb 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments not in sweep config
    parser.add_argument("--dataset_root", type=str, default="../data/UT_HAR_OG", help="Path to the dataset root")
    parser.add_argument("--test_split", type=float, required=True, help="Percentage of data to be used for test")
    parser.add_argument("--val_split", type=float, required=True, help="Percentage of data to be used for validation")
    parser.add_argument("--normalize", action="store_true", help="Normalize the data")
    parser.add_argument("--batch_size_val", type=int, required=True, help="Batch size for validation")

    # Parse arguments
    args = parser.parse_args()

    # Define sweep config
    sweep_config = {
        'method': 'random',  # Search strategy
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'batch_size_train': {'values': [16, 24,  32, 64]},  # Searchable hyperparameter
            'learning_rate': {'min': -5, 'max': -3, 'distribution': 'log_uniform'},
            'embedding_dim': {'values': [128, 256, 512]},  # Searchable hyperparameter
            'num_encoder_layers': {'values': [2, 3, 4, 6, 8]},   # Searchable hyperparameter
            'num_epochs': {'value': 50},                   # Fixed value
            'num_heads': {'values':[4, 8]}, 
            'dropout':{'min':0.1,
                       'max': 0.5},
        }
    }

    # Include arguments not in sweep config as fixed parameters
    sweep_config['parameters']['dataset_root'] = {'value': args.dataset_root}
    sweep_config['parameters']['test_split'] = {'value': args.test_split}
    sweep_config['parameters']['val_split'] = {'value': args.val_split}
    sweep_config['parameters']['normalize'] = {'value': args.normalize}
    sweep_config['parameters']['batch_size_val'] = {'value': args.batch_size_val}

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="HAR-CSI2")
    print(sweep_id)
 