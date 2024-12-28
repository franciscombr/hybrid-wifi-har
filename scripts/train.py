import wandb 
from train_utils_arch02 import wandb_train_sweep 

if __name__ == '__main__':
    sweep_id = "71grhpeu"
    # Run the sweep
    wandb.agent(sweep_id, function=wandb_train_sweep, project = "HAR-CSI2",count=10)