import torch
import matplotlib.pyplot as plt
import seaborn as sns
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
    # text version - rows = actual class, columns = predicted class
    matrix = confusion_matrix(true_labels, predicted_labels)

    print("Confusion matrix")
    print("(rows = actual, columns = predicted)")
    print("Classes:", class_names)
    print(matrix)


def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    class_names,
    title="Confusion matrix",
    ax=None,
    normalize=False,
    cmap="Blues",
):
    """
    Plot a confusion matrix with matplotlib/seaborn.

    ax:        optional matplotlib Axes to draw into (for subplot grids).
               If None, a new figure is created and shown.
    normalize: if True, show row-normalized values (i.e. per-class recall).
               Useful for small datasets where raw counts are hard to compare.
    """
    cm = confusion_matrix(true_labels, predicted_labels)

    fmt = "d"
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"

    show_fig = ax is None
    if show_fig:
        fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
        vmin=0 if normalize else None,
        vmax=1 if normalize else None,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    if show_fig:
        plt.tight_layout()
        plt.show()


def show_classification_report(true_labels, predicted_labels, class_names):
    # precision, recall, f1-score, per class
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    print(report)