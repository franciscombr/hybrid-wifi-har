import argparse
import datetime
import torch
import time
import numpy as np
import wandb

from src.ut_har.ut_har import make_dataset, make_dataloader
from src.resnet.resnet_arch01 import CustomResNet18 


###############################
#
#      HELPER FUNCTIONS
#
###############################
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, name_model="default"):
    """
    Save a checkpoint of the model state, optimizer state, epoch, and loss.

    :param model: Model to save.
    :param optimizer: Optimizer to save.
    :param epoch: Current epoch.
    :param loss: Loss value at this epoch.
    :param checkpoint_dir: Directory to save checkpoints.
    """
    checkpoint_path = f"{checkpoint_dir}checkpoint_epoch_{epoch + 1}_{name_model}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'antennas_num': model.num_antennas,
        'subcarriers_num': model.num_subcarriers,
    }, checkpoint_path)
    print(f"     >> Checkpoint saved at: {checkpoint_path}")


########################################
#
#      TRAINING AND TESTING FUNCTIONS
#
########################################
def print_model_settings(model):
    """
    Prints a summary of the model settings.

    Args:
        model: The model whose settings are to be printed.
    """
    # Extract settings from the model
    settings = {
        "Number of Antennas": model.num_antennas,
        "Number of Subcarriers": model.num_subcarriers,
        "Number of Time Slices": model.num_time_slices,
    }

    # Print settings in a formatted way
    print("[Model Settings Summary]")
    print("-" * 40)
    for key, value in settings.items():
        print(f"  * {key:<30}: {value}")
    print("-" * 40)



def train_epoch(model, device, dataloader, loss_fn, optimizer, epoch):
    model.train()
    train_loss = []
    train_acc = []


    for X, y in dataloader:
        # Extract inputs and outputs from the batch
        wifi_csi_frame = X.to(device)  # Input data
        label = y.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(wifi_csi_frame) 
        loss = loss_fn(outputs,label)
        _, predicted = torch.max(outputs,1)
        acc = (predicted == label).sum() / predicted.size(0)
        loss.backward()
        optimizer.step()

        # Record loss
        train_loss.append(loss.item())
        train_acc.append(acc.item())

    # Compute and print the average loss for the epoch
    avg_loss = np.mean(train_loss)
    avg_acc = np.mean(train_acc)
    print(f"     - Epoch [{epoch + 1}] >> Average Training Loss: {avg_loss:.6f} | Average Training Accuracy: {avg_acc:.6f}")

    return avg_loss, avg_acc


def test_epoch(model, device, dataloader, loss_fn, epoch, dataset_type="Validation"):
    model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for X,y in dataloader:
            # Extract inputs and outputs from the batch
            wifi_csi_frame = X.to(device)
            label = y.to(device)

            # Forward pass
            outputs = model(wifi_csi_frame)

            # Compute loss
            loss = loss_fn(outputs, label)
            val_loss.append(loss.item())

            #Compute Accuracy
            _, predicted = torch.max(outputs,1)
            acc = (predicted == label).sum() / predicted.size(0)
            val_acc.append(acc.item())

    avg_loss = np.mean(val_loss)
    avg_acc = np.mean(val_acc)
    print(f"     >> {dataset_type} loss: {avg_loss:.6f} | acc: {avg_acc:.6f}")
    return avg_loss, avg_acc

def wandb_train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        print(f"Training with the following configuration:")
        print(config)

        # Access all arguments from config
        dataset_root = config.dataset_root
        test_split = config.test_split
        val_split = config.val_split
        normalize = config.normalize
        batch_size_train = config.batch_size_train
        batch_size_val = config.batch_size_val
        num_epochs = config.num_epochs
        learning_rate = config.learning_rate

        train_test(dataset_root, normalize, val_split, test_split, batch_size_train, batch_size_val,  num_epochs, learning_rate)
         
def train_test(dataset_root, normalize, val_split, test_split, batch_size_train, batch_size_val,  num_epochs, learning_rate):

    print('\n')
    print('*******************************************************************************')
    print('*                         Training model                                      *')
    print('*******************************************************************************')
    print('\n')

    date_train = datetime.datetime.now().strftime('%d-%b-%Y_(%H:%M:%S)')
    print(f"  * Date: {date_train}")

    #############################
    #       LOAD DATASET        #
    #############################
    print(f"  * Dataset path: {dataset_root}")

    train_dataset, val_dataset, test_dataset = make_dataset(dataset_root, normalize, val_split, test_split)

    rng_generator = torch.manual_seed(42)
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator,batch_size=batch_size_train)
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=batch_size_val)
    test_loader = make_dataloader(test_dataset, is_training=False, generator=rng_generator, batch_size=batch_size_val)

    print(f"[TRAINING]")
    print(f"    >> Train set samples: {len(train_loader)}. Batch size: {batch_size_train}")
    print(f"    >> Test set samples: {len(val_loader)}")
    print(f"    >> Selected data split: {val_split}")

    #############################
    #        MODEL CONFIG       #
    #############################
    torch.cuda.empty_cache()

    # Initialize model, optimizer, loss function, and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"    >> Training in: {device}")

    model = CustomResNet18(
        num_antennas=3,
        num_subcarriers=30,
        num_time_slices=10,
        num_classes=8,
    ).to(device)

    learnig_rate = learning_rate 
    optimizer_name = "NAdam"
    optimizer = torch.optim.NAdam(model.parameters(), lr=learnig_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    print_model_settings(model)

    num_epochs = num_epochs  # Set your desired number of epochs
    history_da = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    t0 = time.time()

    for epoch in range(num_epochs):
        print('    >> EPOCH %d/%d' % (epoch + 1, num_epochs))
        t1 = time.time()

        # Training
        train_loss, train_acc = train_epoch(
            model=model,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch
        )

        # Scheduler step
        scheduler.step()

        # Optionally, print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    >> Learning Rate after epoch {epoch + 1}: {current_lr:.6e}")

        #Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "learning_rate": current_lr
        })

        # Validation
        val_loss, val_acc = test_epoch(
            model=model,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn,
            epoch=epoch + 1,
            dataset_type="Validation",
        )

        # Log validation metrics to WandB
        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Record losses
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        history_da['train_acc'].append(train_acc)
        history_da['val_acc'].append(val_acc)

        # Print epoch summary
        print('    >> EPOCH {}/{} \t train loss {:.6f} \t train_acc {:.6f} \t val loss {:.6f} \t val acc {:.6f}'.format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

        # Save checkpoint at the end of each epoch
        model_version = "arch02_v1_001"
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir="../results/models/checkpoints/", name_model=model_version)

        print(f"    >> Consumed time in Epoch {epoch + 1}: {time.time() - t1:.2f} seconds \n")

        # Saving the model.
    save_model = True
    if save_model:
        model_version = "arch01_v2_001"
        day_model = "091024"
        torch.save(model.state_dict(), f"../results/models/hybrid_har_model_{day_model}{model_version}_ep{num_epochs}_lr{learnig_rate}_{optimizer_name}.pth")
        print(f" >> Model saved with name: hybrid_har_model_{day_model}{model_version}_ep{num_epochs}_lr{learnig_rate}_{optimizer_name}.pth")

    print(f"Total training time: {(time.time() - t0) / 60:.2f} minutes")

def main(args):
    # Initialize WandB
    wandb.init(
        project="HAR-CSI2",
        config={
            "dataset_root": args.dataset_root,
            "batch_size_train": args.batch_size_train,
            "batch_size_val": args.batch_size_val,
            "test_split": args.test_split,
            "val_split": args.val_split,
            "normalize": args.normalize,
            "learning_rate": args.learning_rate,
            "optimizer": "NAdam",
            "num_epochs": args.num_epochs,
            "num_classes": 8,
            "arch": "resnet"
        },
        name=f"CSI2HAR_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    config = wandb.config  # Access hyperparameters

    # Access all arguments from config
    dataset_root = config.dataset_root
    test_split = config.test_split
    val_split = config.val_split
    normalize = config.normalize
    batch_size_train = config.batch_size_train
    batch_size_val = config.batch_size_val
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate

    train_test(dataset_root, normalize, val_split, test_split, batch_size_train, batch_size_val,  num_epochs, learning_rate)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model in the HAR dataset.")

    parser.add_argument("--dataset_root", type=str, default="../data/UT_HAR_OG",
                        help="Path to the dataset root")
    parser.add_argument("--test_split", type=float, required=True,
                        help="Percentage of data to be used for test")
    parser.add_argument("--val_split", type=float, required=True, 
                        help="Percentage of data to be used for validation")
    parser.add_argument("--normalize", action="store_true", 
                        help="Choose to normalize the data to zero mean and unit variance")
    parser.add_argument("--batch_size_train", type=int, required=True,
                        help="Choose an adequate batch size for model training.")
    parser.add_argument("--batch_size_val", type=int, required=True,
                        help="Choose an adequate batch size for model validation.")
    parser.add_argument("--num_epochs", type=int, default=30, 
                        help="Choose number of epochs for model training")
    parser.add_argument("--learning_rate", type=float, default=1e-4 )
    args = parser.parse_args()
    
    main(args)
