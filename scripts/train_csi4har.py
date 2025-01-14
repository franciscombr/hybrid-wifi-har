import wandb 
from src.csi2har.train_utils_arch01 import wandb_train_sweep 

if __name__ == '__main__':
    sweep_id = "5fdmh18b"
    # Run the sweep
    wandb.agent(sweep_id, function=wandb_train_sweep, project = "HAR-CSI2",count=10)