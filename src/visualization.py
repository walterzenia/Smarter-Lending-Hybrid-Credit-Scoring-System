"""Small visualization stubs for plotting model results."""
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, labels=None, figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    return fig
