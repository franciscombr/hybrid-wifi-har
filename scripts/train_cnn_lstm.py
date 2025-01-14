import wandb 
from src.cnn_lstm.train_utils_cnn_lstm import wandb_train_sweep 

if __name__ == '__main__':
    sweep_id = "5xv62wto"
    # Run the sweep
    wandb.agent(sweep_id, function=wandb_train_sweep, project = "HAR-CSI2",count=10)