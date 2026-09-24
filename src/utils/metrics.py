import torch
from sklearn.metrics import confusion_matrix, classification_report

def get_predictions(model, data_loader, device):
    # runs the model on a full dataset and collects all predictions + true labels
    model.eval()
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_predicted


def show_confusion_matrix(true_labels, predicted_labels, class_names):
    # rows = actual class, columns = predicted class
    # a healthy model has big numbers on the diagonal, small numbers elsewhere
    matrix = confusion_matrix(true_labels, predicted_labels)

    print("Confusion matrix")
    print("(rows = actual, columns = predicted)")
    print("Classes:", class_names)
    print(matrix)


def show_classification_report(true_labels, predicted_labels, class_names):
    # precision, recall, f1-score, per class
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    print(report)