"""
train_model.py
---------------
Train a Convolutional Neural Network (CNN) on the MNIST dataset
to recognize handwritten digits (0-9).

Outputs:
    - digit_model.h5         : Trained Keras model
    - training_history.png   : Accuracy & loss curves
    - confusion_matrix.png   : Confusion matrix on test set
    - classification_report.txt
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)   # (N, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion(y_true, y_pred, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix - MNIST Test Set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    print("[1/4] Loading MNIST dataset...")
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_data()
    print(f"   Train: {x_train.shape}, Test: {x_test.shape}")

    print("[2/4] Building CNN model...")
    model = build_model()
    model.summary()

    print("[3/4] Training model...")
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        x_train, y_train_cat,
        validation_split=0.1,
        epochs=12,
        batch_size=128,
        callbacks=[es],
        verbose=2,
    )

    print("[4/4] Evaluating model...")
    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"   Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    plot_history(history)
    plot_confusion(y_test, y_pred)

    report = classification_report(y_test, y_pred, digits=4)
    with open("classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {acc*100:.2f}%\nTest Loss: {loss:.4f}\n\n{report}")
    print(report)

    model.save("digit_model.h5")
    print("Saved -> digit_model.h5")


if __name__ == "__main__":
    main()
