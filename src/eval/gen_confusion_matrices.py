from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(true_labels, predicted_labels, model_name, normalize='true', cmap='viridis'):
    """
    Plots a confusion matrix with class names instead of numbers.
    
    Parameters:
    - true_labels (list or array): Ground truth labels.
    - predicted_labels (list or array): Predicted labels from the model.
    - label_encoder (dict): Dictionary mapping class names to numeric labels.
    - model_name (str): Name of the model for title.
    - normalize (str, optional): 'true', 'pred', or None. Normalizes confusion matrix if set.
    - cmap (str, optional): Colormap for the confusion matrix.
    """
    # Decode numeric labels to class names
    label_encoder = {"NoActivity": 0, "bed": 1, "fall": 2, "pickup": 3, "run": 4, "sitdown": 5, "standup": 6, "walk": 7}
    class_names = {v: k for k, v in label_encoder.items()}  # Reverse the label encoder
    decoded_true_labels = [class_names[label] for label in true_labels]
    decoded_predicted_labels = [class_names[label] for label in predicted_labels]
    sorted_class_names = sorted(class_names.values())
    # Generate the confusion matrix
    cm = confusion_matrix(decoded_true_labels, decoded_predicted_labels, labels=sorted_class_names, normalize=normalize)
    
    # Create the display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
    disp.plot(cmap='Blues', xticks_rotation='vertical', colorbar=False)  # Disable colorbar
    file_path = f"{model_name}_conf_matrix.png"
    plt.savefig(file_path, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    # Example usage
    # Replace these lists with your true labels and predictions for each model
    true_labels = [0, 1, 2, 3, 4, 5, 6, 7]  # Ground truth numeric labels
    predicted_labels_model_1 = [0, 1, 2, 3, 4, 5, 6, 7]  # Predictions from Model 1
    predicted_labels_model_2 = [0, 1, 2, 2, 4, 5, 6, 7]  # Predictions from Model 2

    # Call the function for each model
    plot_confusion_matrix(true_labels, predicted_labels_model_1, model_name='Model 1')
    plot_confusion_matrix(true_labels, predicted_labels_model_2, model_name='Model 2')