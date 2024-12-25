import argparse
import datetime
import torch
import time
import numpy as np

from src.ut_har.ut_har import make_dataset, make_dataloader


###############################
#
#      HELPER FUNCTIONS
#
###############################
#TODO: change model parameters to be saved according to model architecture
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
        'embeddings': model.embedding_dim,
        'head_num': model.num_heads,
        'antennas_num': model.num_antennas,
        'subcarriers_num': model.num_subcarriers,
        'cloud_points_num': model.num_points,
    }, checkpoint_path)
    print(f"     >> Checkpoint saved at: {checkpoint_path}")


########################################
#
#      TRAINING AND TESTING FUNCTIONS
#
########################################
#TODO: change model parameters to be saved according to model architecture
def print_model_settings(model):
    """
    Prints a summary of the model settings.

    Args:
        model: The model whose settings are to be printed.
    """
    # Extract settings from the model
    settings = {
        "Embedding Dimension": model.embedding_dim,
        "Number of Points": model.num_points,
        "Number of Antennas": model.num_antennas,
        "Number of Subcarriers": model.num_subcarriers,
        "Number of Time Slices": model.num_time_slices,
        "Number of Heads": model.num_heads,
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

    # Get the total number of batches for progress tracking
    total_batches = len(dataloader)

    for X, y in dataloader:
        # Extract inputs and outputs from the batch
        wifi_csi_frame = X.to(device)  # Input data
        label = y.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(wifi_csi_frame) 
        loss = loss_fn(outputs,label)

        loss.backward()
        optimizer.step()

        # Record loss
        train_loss.append(loss.item())

    # Compute and print the average loss for the epoch
    avg_loss = np.mean(train_loss)
    print(f"     - Epoch [{epoch + 1}] >> Average Training Loss: {avg_loss:.6f}")

    return avg_loss


def test_epoch(model, device, dataloader, loss_fn, epoch, dataset_type="Validation", visualize=True):
    model.eval()
    val_loss = []
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

    avg_loss = np.mean(val_loss)
    print(f"     >> {dataset_type} loss: {avg_loss:.6f}")
    return avg_loss


def main(args):
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
    dataset_root = args.dataset_root
    print(f"  * Dataset path: {dataset_root}")

    train_dataset, val_dataset, test_dataset = make_dataset(dataset_root, args.normalize, args.val_split, args.test_split)

    rng_generator = torch.manual_seed(42)
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator)
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator)
    test_loader = make_dataloader(test_dataset, is_training=False, generator=rng_generator)

    print(f"[TRAINING]")
    print(f"    >> Train set samples: {len(train_loader)}. Batch size: {args.batch_size}")
    print(f"    >> Test set samples: {len(val_loader)}")
    print(f"    >> Selected data split: {args.test_split}")

    #############################
    #        MODEL CONFIG       #
    #############################
    torch.cuda.empty_cache()

    # Initialize model, optimizer, loss function, and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CSI2PointCloudModel(
        embedding_dim=256,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_points=1200,  # Updated to match the fixed number of points
        num_antennas=3,
        num_subcarriers=114,
        num_time_slices=10
    ).to(device)

    learnig_rate = 1e-4
    optimizer_name = "NAdam"
    optimizer = torch.optim.NAdam(model.parameters(), lr=learnig_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    print_model_settings(model)

    num_epochs = args.num_epochs  # Set your desired number of epochs
    history_da = {'train_loss': [], 'val_loss': []}
    t0 = time.time()

    for epoch in range(num_epochs):
        print('    >> EPOCH %d/%d' % (epoch + 1, num_epochs))
        t1 = time.time()

        # Training
        train_loss = train_epoch(
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

        # Validation
        val_loss = test_epoch(
            model=model,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn,
            epoch=epoch + 1,
            dataset_type="Validation",
            visualize=True
        )

        # Record losses
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)

        # Print epoch summary
        print('    >> EPOCH {}/{} \t train loss {:.6f} \t val loss {:.6f}'.format(epoch + 1, num_epochs, train_loss, val_loss))

        # Save checkpoint at the end of each epoch
        model_version = "arch01_v1_001"
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir="../models/checkpoints/", name_model=model_version)

        print(f"    >> Consumed time in Epoch {epoch + 1}: {time.time() - t1:.2f} seconds \n")

        # Saving the model.
    save_model = True
    if save_model:
        model_version = "arch01_v1_001"
        day_model = "091024"
        torch.save(model.state_dict(), f"../models/hybrid_har_model_{day_model}{model_version}_ep{num_epochs}_lr{learnig_rate}_{optimizer_name}.pth")
        print(f" >> Model saved with name: hybrid_har_model_{day_model}{model_version}_ep{num_epochs}_lr{learnig_rate}_{optimizer_name}.pth")

    print(f"Total training time: {(time.time() - t0) / 60:.2f} minutes")


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
    parser.add_argument("--batch_size", type=float, required=True,
                        help="Choose an adequate batch size for model training.")
    parser.add_argument("--num_epochs", type=float, default=30, 
                        help="Choose number of epochs for model training")
    
    args = parser.parse_args()

    main(args)