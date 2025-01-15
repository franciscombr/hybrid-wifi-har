import torch
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from src.resnet.resnet_arch01 import CustomResNet18  # Import the model class
from src.ut_har.ut_har import make_dataset, make_dataloader
from torch.utils.data import ConcatDataset

# Load the model and checkpoint
checkpoint_path = "path_to_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomResNet18(
    num_antennas=3,
    num_subcarriers=30,
    num_time_slices=10,
    num_classes=8,
).to(device)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode
model.resnet18.fc = torch.nn.Identity()

dataset_root = '/nas-ctm01/datasets/public/CSI-HAR/Dataset/UT_HAR_CAL_PHASE' 
normalize = True
val_split = 0.2
test_split = 0.2
train_full = True
batch_size_train = 16
batch_size_val = 8
train_dataset, val_dataset, test_dataset = make_dataset(dataset_root, normalize, val_split, test_split)
if train_full == True:
    train_dataset = ConcatDataset([train_dataset, val_dataset])
    val_dataset = test_dataset

rng_generator = torch.manual_seed(42)
train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator,batch_size=batch_size_train)
val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=batch_size_val)

# Function to extract features from the second-to-last layer
def extract_features(model, data_loader):
    features_list = []
    labels_list = []
    with torch.no_grad():
        # Forward pass up to the second-to-last layer using hooks
        for X, y in data_loader:
        # Extract inputs and outputs from the batch
            wifi_csi_frame = X.to(device)  # Input data

            # Zero the parameter gradients

            features_batch = model(wifi_csi_frame) 
            features_list.append(features_batch.cpu().numpy())
            labels_list.append(y.numpy())
     
    return np.vstack(features_list), np.hstack(labels_list)
    
# Extract features from the second-to-last layer
X_train_features = extract_features(model, train_loader).cpu().numpy()
X_test_features = extract_features(model, val_loader).cpu().numpy()