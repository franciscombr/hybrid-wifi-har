import wandb 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parse arguments
    args = parser.parse_args()

    # Define sweep config
    sweep_config = {
        'method': 'random',  # Search strategy
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'batch_size_train': {'values': [16, 32,]},  # Searchable hyperparameter
            'learning_rate': {'values': [1e-5, 1e-4]},
            'num_epochs': {'value': 50},                   # Fixed value
            'arch': {'value':'bilstm'},
            'hidden_dim': {'values': [64, 128, 256]},
            'num_layers': {'values': [ 2, 4]},
            'bidirectional': {'value': True},
            'dataset_root':{'values': ['/nas-ctm01/datasets/public/CSI-HAR/Dataset/UT_HAR_OG',
                                        '/nas-ctm01/datasets/public/CSI-HAR/Dataset/UT_HAR_CSI_RATIO',
                                        '/nas-ctm01/datasets/public/CSI-HAR/Dataset/UT_HAR_CAL_PHASE']},

        }
    }

    # Include arguments not in sweep config as fixed parameters
    sweep_config['parameters']['test_split'] = {'value':0.2}
    sweep_config['parameters']['val_split'] = {'value':0.2}
    sweep_config['parameters']['normalize'] = {'value':True}
    sweep_config['parameters']['batch_size_val'] = {'value' : 8}

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="HAR-CSI2")
    print(sweep_config['parameters']['arch'], ": ", sweep_id)


 