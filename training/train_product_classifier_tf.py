"""
train_product_classifier_tf.py -- OPTIONAL stretch-goal upgrade to
train_product_classifier.py.

Trains a MobileNetV2 transfer-learning classifier with TensorFlow/Keras
for higher accuracy than the default HOG+SVM baseline, once you have a
real labeled image dataset.

Requires: pip install tensorflow (not installed by default).
Point DATA_DIR at data/product_images/<class_name>/*.jpg (one folder
per category), then run this script. The saved model is auto-detected
by app/services/cv_service.py (bundle["kind"] == "keras") -- no other
code changes needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config

DATA_DIR = config.DATA_DIR / "product_images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def main():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        raise SystemExit(
            "TensorFlow is not installed. Run `pip install tensorflow` first, "
            "or use the default training/train_product_classifier.py instead."
        )

    if not DATA_DIR.exists():
        raise SystemExit(
            f"Expected a labeled image dataset at {DATA_DIR} "
            "(one subfolder per class). See this file's docstring."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    print("Classes:", class_names)

    normalization = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    keras_path = config.MODELS_DIR / "product_classifier_mobilenetv2.h5"
    model.save(keras_path)

    import joblib
    joblib.dump(
        {"kind": "keras", "keras_path": str(keras_path), "class_names": class_names},
        config.PRODUCT_CLASSIFIER_PATH,
    )
    print(f"Saved MobileNetV2 model to {keras_path} and bundle to {config.PRODUCT_CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
