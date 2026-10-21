import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history: dict):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, history["train_loss"], label="train")

    ax.plot(epochs, history["val_loss"], label="validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("Loss")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()


def plot_metrics(history: dict):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, history["train_f1"], label="train F1")
    ax.plot(epochs, history["val_f1"], label="validation F1")

    ax.plot(
        epochs,
        history["train_precision"],
        "--",
        label="train precision",
    )

    ax.plot(
        epochs,
        history["val_precision"],
        "--",
        label="validation precision",
    )

    ax.plot(
        epochs,
        history["train_recall"],
        ":",
        label="train recall",
    )

    ax.plot(
        epochs,
        history["val_recall"],
        ":",
        label="validation recall",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Model metrics")

    ax.set_ylim(0.0, 1.0)

    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    plt.show()
