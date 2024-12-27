from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import h5py
import torch

class HARData(Dataset):
    def __init__(self,X,y):
        super().__init__()
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.uint8)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def load_data(dataset_path):
    with h5py.File(dataset_path + "/X.h5", "r") as f:
        X = torch.tensor(f["X"][:])

    with h5py.File(dataset_path + "/y.h5", "r") as f:
        y = torch.tensor(f["y"][:])

    return X, y

def make_dataset(dataset_path, normalize, val_split, test_split):
    X, y = load_data(dataset_path)
    
    if test_split > 0:
        # Split into training+validation and test datasets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_split, stratify=y if len(set(y)) > 1 else None, random_state=42
        )
    else:
        # No test dataset, use the entire data for training and validation
        X_train_val, X_test, y_train_val, y_test = X, None, y, None
    
    # Split training+validation dataset into training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split, stratify=y_train_val if len(set(y_train_val)) > 1 else None, random_state=42
    )
    
    if normalize:
        global_mean = X_train.mean(axis=(0, 1))
        global_std = X_train.std(axis=(0, 1)) + 1e-6  # Avoid division by zero
        
        X_train = (X_train - global_mean) / global_std
        X_val = (X_val - global_mean) / global_std
        if X_test is not None:
            X_test = (X_test - global_mean) / global_std
    
    # Create datasets
    train_dataset = HARData(X_train, y_train)
    val_dataset = HARData(X_val, y_val)
    test_dataset = HARData(X_test, y_test) if X_test is not None else None
    
    return train_dataset, val_dataset, test_dataset

def make_dataloader(dataset, is_training, batch_size,generator ):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        generator=generator,
        drop_last=True
    )
    return loader 