import matplotlib.pyplot as plt
import numpy as np


def plot_toxicity_distribution(train_dataset, test_dataset, val_dataset=None):
    train_y = [g.y.item() for g in train_dataset]
    test_y = [g.y.item() for g in test_dataset]
    val_y = [g.y.item() for g in val_dataset] if val_dataset is not None else None

    plt.figure(figsize=(8, 4))
    plt.hist(train_y, bins=50, alpha=0.5, label="Train", density=True)
    if val_y is not None:
        plt.hist(val_y, bins=50, alpha=0.5, label="Val", density=True)
    plt.hist(test_y, bins=50, alpha=0.5, label="Test", density=True)
    plt.xlabel("log10c")
    plt.legend()
    title = "Target distribution: train vs val vs test" if val_y is not None else "Target distribution: train vs test"
    plt.title(title)
    plt.show()

    print(f"Train mean: {np.mean(train_y):.2f}, std: {np.std(train_y):.2f}")
    if val_y is not None:
        print(f"Val mean:   {np.mean(val_y):.2f}, std: {np.std(val_y):.2f}")
    print(f"Test mean:  {np.mean(test_y):.2f},  std: {np.std(test_y):.2f}")


def plot_training(history, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.plot(history["train_loss"], label="Train Loss")

    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="Val Loss")
    if history.get("test_loss"):
        plt.plot(history["test_loss"], label="Test Loss")

    best_epoch = history.get("best_epoch")
    if best_epoch is not None:
        plt.axvline(best_epoch, color="k", linestyle="--", alpha=0.4, label="Best Epoch")

    title = "Training Loss"
    if history.get("val_loss") and history.get("test_loss"):
        title = "Training, Validation, and Test Loss"
    elif history.get("val_loss"):
        title = "Training and Validation Loss"
    elif history.get("test_loss"):
        title = "Training and Test Loss"

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
