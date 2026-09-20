"""
skin_disease_classification.py
--------------------------------
AI-based Skin Disease Classification — Mini Project (single-file version)

Approach: Transfer learning with MobileNetV2 (pretrained on ImageNet),
fine-tuned on a labeled skin-lesion image dataset.

Expected dataset layout (one folder per class):

    dataset/
        train/
            melanoma/*.jpg
            nevus/*.jpg
            bcc/*.jpg
            ...
        val/
            melanoma/*.jpg
            ...
        test/
            melanoma/*.jpg
            ...

Run:
    python skin_disease_classification.py --data_dir dataset --epochs 15
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image as keras_image

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
def get_generators(data_dir, img_size=224, batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_dir = os.path.join(data_dir, "test")
    test_gen = None
    if os.path.isdir(test_dir):
        test_gen = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
    return train_gen, val_gen, test_gen


# ----------------------------------------------------------------------
# 2. MODEL
# ----------------------------------------------------------------------
def build_model(num_classes, img_size=224):
    inputs = Input(shape=(img_size, img_size, 3))
    base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
    base_model.trainable = False  # freeze for phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base_model


def unfreeze_for_finetuning(model, base_model, num_layers=30, lr=1e-5):
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------------------------------------------------
# 3. TRAIN
# ----------------------------------------------------------------------
def train(args):
    os.makedirs("saved_model", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_gen, val_gen, test_gen = get_generators(args.data_dir, args.img_size, args.batch_size)
    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    with open("saved_model/class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    weights = compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
    class_weights = dict(enumerate(weights))

    model, base_model = build_model(num_classes, args.img_size)

    ckpt_path = "saved_model/skin_disease_model.h5"
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    print("\n=== Phase 1: training classifier head (base frozen) ===")
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=max(1, args.epochs // 2),
                    class_weight=class_weights, callbacks=callbacks)

    print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
    model = unfreeze_for_finetuning(model, base_model)
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs - args.epochs // 2,
                    class_weight=class_weights, callbacks=callbacks)

    model.save(ckpt_path)
    print(f"\nModel saved to {ckpt_path}")

    # Plot curves
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(acc, label="Train"); axes[0].plot(val_acc, label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(loss, label="Train"); axes[1].plot(val_loss, label="Val")
    axes[1].set_title("Loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png")
    print("Saved outputs/training_curves.png")

    if test_gen is not None:
        evaluate(model, test_gen, class_names)


# ----------------------------------------------------------------------
# 4. EVALUATE
# ----------------------------------------------------------------------
def evaluate(model, test_gen, class_names):
    print("\n=== Evaluating on test set ===")
    probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("Saved outputs/confusion_matrix.png")


# ----------------------------------------------------------------------
# 5. PREDICT A SINGLE IMAGE
# ----------------------------------------------------------------------
def predict_image(img_path, model_path="saved_model/skin_disease_model.h5",
                   class_indices_path="saved_model/class_indices.json", img_size=224):
    model = load_model(model_path)
    with open(class_indices_path) as f:
        class_indices = json.load(f)
    class_names = [None] * len(class_indices)
    for name, idx in class_indices.items():
        class_names[idx] = name

    img = keras_image.load_img(img_path, target_size=(img_size, img_size))
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    top = int(np.argmax(preds))
    print(f"\nPredicted: {class_names[top]}  ({preds[top]*100:.2f}% confidence)")
    for name, p in sorted(zip(class_names, preds), key=lambda x: -x[1]):
        print(f"  {name:25s} {p*100:6.2f}%")


# ----------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-based skin disease classification")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--image", type=str, help="Path to image (mode=predict)")
    parser.add_argument("--model_path", type=str, default="saved_model/skin_disease_model.h5")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if not args.image:
            raise ValueError("--image is required when --mode predict")
        predict_image(args.image, model_path=args.model_path, img_size=args.img_size)
