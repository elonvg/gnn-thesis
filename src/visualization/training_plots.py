import matplotlib.pyplot as plt
import numpy as np


def plot_toxicity_distribution(train_dataset, test_dataset):
    train_y = [g.y.item() for g in train_dataset]
    test_y = [g.y.item() for g in test_dataset]

    plt.figure(figsize=(8, 4))
    plt.hist(train_y, bins=50, alpha=0.5, label="Train", density=True)
    plt.hist(test_y, bins=50, alpha=0.5, label="Test", density=True)
    plt.xlabel("log10c")
    plt.legend()
    plt.title("Target distribution: train vs test")
    plt.show()

    print(f"Train mean: {np.mean(train_y):.2f}, std: {np.std(train_y):.2f}")
    print(f"Test mean:  {np.mean(test_y):.2f},  std: {np.std(test_y):.2f}")


def plot_training(history, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title("Training and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
