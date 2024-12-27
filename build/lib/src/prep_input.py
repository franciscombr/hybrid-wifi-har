import h5py 
import torch

#Load data
def load_data(dataset_path):
    with h5py.File(dataset_path + "/data/X.h5", "r") as f:
        X = torch.tensor(f["X"][:])

    with h5py.File(dataset_path + "/label/y.h5", "r") as f:
        y = torch.tensor(f["y"][:])

    return X,y

def main():
    dataset_path = "../data/UT_HAR_OG"
    X, y = load_data(dataset_path)
    