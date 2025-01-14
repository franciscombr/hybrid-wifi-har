import wandb 
from src.lstm.train_utils_lstm import wandb_train_sweep 

if __name__ == '__main__':
    sweep_id = "uwuer8b7"
    # Run the sweep
    wandb.agent(sweep_id, function=wandb_train_sweep, project = "HAR-CSI2",count=10)