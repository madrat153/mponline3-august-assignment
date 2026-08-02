"""
OPTIONAL upgrade path for Module A2: real transfer learning with
MobileNetV2, exactly as described in the project brief, against a real
labeled image folder.

This script is NOT required to run the platform -- the default
scikit-learn/HOG pipeline (train_image_classifier.py) already produces a
working product_classifier.joblib. Use this once you have:
  1. `pip install tensorflow` (or tensorflow-cpu)
  2. A real dataset laid out as:
       data/product_images/
         clothing/*.jpg
         shoes/*.jpg
         bags/*.jpg
         electronics/*.jpg
         groceries/*.jpg
     (e.g. a curated subset of Fashion-MNIST or the Kaggle "Retail Product
     Checkout Dataset" referenced in the project brief.)

Usage:
    python training/train_image_classifier_tf.py --data-dir data/product_images
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PRODUCT_CATEGORIES, MODELS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/product_images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import MobileNetV2
    except ImportError:
        print(
            "TensorFlow is not installed. Run `pip install tensorflow` to use this "
            "upgrade path, or stick with training/train_image_classifier.py."
        )
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Dataset folder {data_dir} not found. See the module docstring for the expected layout.")
        sys.exit(1)

    img_size = (args.img_size, args.img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=42,
        image_size=img_size, batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=img_size, batch_size=args.batch_size,
    )
    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    normalization = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)

    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False  # transfer learning: freeze the pretrained backbone

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    out_path = MODELS_DIR / "product_classifier.h5"
    model.save(out_path)
    print(f"Saved MobileNetV2 model -> {out_path}")
    print(
        "NOTE: app/services/cv_service.py currently loads the scikit-learn "
        "bundle by default. To serve this .h5 model instead, add a small "
        "TensorFlow-backed branch to ProductClassifierService (set "
        "PRODUCT_CLASSIFIER_BACKEND=tensorflow) that calls model.predict() "
        "on a normalized (img_size, img_size, 3) batch and maps the softmax "
        "output back onto `class_names`."
    )


if __name__ == "__main__":
    main()
