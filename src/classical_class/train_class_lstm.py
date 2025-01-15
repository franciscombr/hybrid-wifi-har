import torch
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from src.lstm.lstm_arch01 import LSTMBasedModel
from src.ut_har.ut_har import make_dataset, make_dataloader
from torch.utils.data import ConcatDataset
from src.eval.gen_confusion_matrices import plot_confusion_matrix

# Load the model and checkpoint
checkpoint_path = "../../results/models/checkpoints/best_model_CSI2HAR_2025-01-15_15-18-09_6ggpb683.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMBasedModel(
        num_antennas=3,
        num_subcarriers=30,
        num_time_slices=10,
        num_classes=8,
        hidden_dim=256,
        num_layers=2,
        bidirectional=True,
    ).to(device)


checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode
model.fc = torch.nn.Identity()

dataset_root = '/nas-ctm01/datasets/public/CSI-HAR/Dataset/UT_HAR_CAL_PHASE' 
#dataset_root = '/home/francisco.m.ribeiro/PDEEC/ML/Project/hybrid-wifi-har/data/UT_HAR_CAL_PHASE' #'/nas-ctm01/datasets/public/CSI-HAR/Dataset/UT_HAR_CAL_PHASE' 
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
X_train_features, y_train_labels= extract_features(model, train_loader)
X_test_features, y_test_labels = extract_features(model, val_loader)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Train the SVM classifier
svm_classifier = LinearSVC( random_state=42, multi_class="crammer_singer")
svm_classifier.fit(X_train_scaled, y_train_labels)

# Evaluate the classifier
y_pred = svm_classifier.predict(X_test_scaled)
print("SVM Classification Report:")
print(classification_report(y_test_labels, y_pred))
print(f"Accuracy: {accuracy_score(y_test_labels, y_pred):.2f}")
plot_confusion_matrix(y_test_labels, y_pred,'linearsvm',normalize='pred')
# Train and evaluate Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train_labels)
y_pred_nb = nb_classifier.predict(X_test_scaled)
print("Naive Bayes Classification Report:")
print(classification_report(y_test_labels, y_pred_nb))
print(f"Naive Bayes Accuracy: {accuracy_score(y_test_labels, y_pred_nb):.2f}")
plot_confusion_matrix(y_test_labels, y_pred_nb,'gaussianNB',normalize='pred')

# Train and evaluate K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=6)
knn_classifier.fit(X_train_scaled, y_train_labels)
y_pred_knn = knn_classifier.predict(X_test_scaled)
print("K-Nearest Neighbors Classification Report:")
print(classification_report(y_test_labels, y_pred_knn))
print(f"K-Nearest Neighbors Accuracy: {accuracy_score(y_test_labels, y_pred_knn):.2f}")
plot_confusion_matrix(y_test_labels,y_pred_knn,'knn5',normalize='pred')