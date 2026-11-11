import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
import tarfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models, optimizers
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 2            # phase 1: train only the new classification layers
EPOCHS_FINE = 6            # phase 2: fine-tune the pretrained network
SUBSET_SIZE = 750          # training images per class (Food-101 has 750)
VAL_SPLIT = 0.1
UNFREEZE_LAST_LAYERS = 150  # layers of the backbone tuned in phase 2
HEAD_LR = 1e-3
FINE_TUNE_LR = 2e-4        # starting rate in phase 2, decays smoothly towards zero
LABEL_SMOOTHING = 0.1
CONFIDENCE_THRESHOLD = 0.40
TOP_K = 5

DATASET_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
DATASET_DIR = Path("dataset")
RAW_DIR = DATASET_DIR / "food-101"
IMAGES_DIR = RAW_DIR / "images"
META_DIR = RAW_DIR / "meta"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

BEST_MODEL_PATH = MODELS_DIR / "plateiq_best.keras"
CLASSES_PATH = MODELS_DIR / "classes.json"
HISTORY_PATH = MODELS_DIR / "history.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ACCURACY_PLOT_PATH = RESULTS_DIR / "accuracy_plot.png"
LOSS_PLOT_PATH = RESULTS_DIR / "loss_plot.png"
PER_CLASS_PLOT_PATH = RESULTS_DIR / "per_class_f1.png"
CLASS_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
METRICS_PATH = RESULTS_DIR / "metrics.json"

TRAIN_PATHS_FILE = DATASET_DIR / "train_paths.npy"
TRAIN_LABELS_FILE = DATASET_DIR / "train_labels.npy"
TEST_PATHS_FILE = DATASET_DIR / "test_paths.npy"
TEST_LABELS_FILE = DATASET_DIR / "test_labels.npy"

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n {title}\n{'=' * 60}")


def print_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def print_error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def print_success(msg: str) -> None:
    print(f"[SUCCESS] {msg}")


def ensure_dirs() -> None:
    for d in (DATASET_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def display_name(raw: str) -> str:
    """Turn 'french_fries' into 'French Fries'."""
    return raw.replace("_", " ").title()


def read_classes_file() -> List[str]:
    classes_file = META_DIR / "classes.txt"
    if not classes_file.exists():
        return []
    return [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]


def load_class_names() -> List[str]:
    """Class names used by the trained model (falls back to the dataset list)."""
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH) as f:
            return json.load(f)
    return read_classes_file()


def check_gpu() -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print_info(f"GPU available: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print_error(f"GPU memory growth error: {e}")
        return True
    print_info("No GPU found. Running on CPU (training will be slow).")
    return False


def verify_imports() -> None:
    print_info(f"TensorFlow: {tf.__version__}")
    print_info(f"NumPy: {np.__version__}")
    print_info(f"Pandas: {pd.__version__}")


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def dataset_ready() -> bool:
    if not (META_DIR / "train.txt").exists() or not (META_DIR / "test.txt").exists():
        return False
    classes = read_classes_file()
    if not classes:
        return False
    return all((IMAGES_DIR / c).is_dir() for c in classes)


def download_dataset() -> bool:
    ensure_dirs()
    archive_path = DATASET_DIR / "food-101.tar.gz"
    part_path = DATASET_DIR / "food-101.tar.gz.part"

    if dataset_ready():
        print_info("Dataset already extracted. Skipping download.")
        return True

    if archive_path.exists():
        print_info("Archive already downloaded.")
    else:
        print_info(f"Downloading Food-101 (about 5 GB) from {DATASET_URL}")
        try:
            with requests.get(DATASET_URL, stream=True, timeout=60) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with open(part_path, "wb") as f, tqdm(
                    desc="Downloading", total=total, unit="B",
                    unit_scale=True, unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            part_path.replace(archive_path)  # rename only when fully downloaded
            print_success("Download complete")
        except Exception as e:
            print_error(f"Download failed: {e}")
            return False

    print_info("Extracting dataset (this can take several minutes)...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tqdm(tar, desc="Extracting", unit="file"):
                if member.isfile() and (DATASET_DIR / member.name).exists():
                    continue  # already extracted earlier
                if hasattr(tarfile, "data_filter"):
                    tar.extract(member, path=DATASET_DIR, filter="data")
                else:
                    tar.extract(member, path=DATASET_DIR)
    except Exception as e:
        print_error(f"Extraction failed: {e}")
        return False

    if not dataset_ready():
        print_error("Extraction finished but expected folders are missing.")
        return False
    print_success("Extraction complete")
    return True


def collect_split(
    lines: List[str], classes: List[str], limit: int = 0
) -> Tuple[List[str], List[int], int]:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    per_class: Dict[str, List[str]] = {c: [] for c in classes}
    for line in lines:
        cls = line.split("/")[0]
        if cls in per_class:
            per_class[cls].append(line)

    paths, labels, missing = [], [], 0
    for cls in classes:
        items = per_class[cls][:limit] if limit else per_class[cls]
        for rel in items:
            p = IMAGES_DIR / f"{rel}.jpg"
            if p.exists():
                paths.append(str(p))
                labels.append(class_to_idx[cls])
            else:
                missing += 1
    return paths, labels, missing


def prepare_dataset(subset_size: int = SUBSET_SIZE) -> bool:
    print_section("DATASET PREPARATION")
    ensure_dirs()

    if not download_dataset():
        return False

    classes = read_classes_file()
    train_file = META_DIR / "train.txt"
    test_file = META_DIR / "test.txt"
    train_lines = [l.strip() for l in train_file.read_text().splitlines() if l.strip()]
    test_lines = [l.strip() for l in test_file.read_text().splitlines() if l.strip()]

    train_paths, train_labels, miss_a = collect_split(train_lines, classes, subset_size)
    test_paths, test_labels, miss_b = collect_split(test_lines, classes)

    if miss_a or miss_b:
        print_error(
            f"{miss_a + miss_b} image files are missing. Delete the folder "
            f"'{RAW_DIR}' and run --prepare-data again to re-extract."
        )
        return False

    print_info(f"Food categories: {len(classes)}")
    print_info(f"Training images: {len(train_paths)}")
    print_info(f"Test images: {len(test_paths)}")

    np.save(TRAIN_PATHS_FILE, np.array(train_paths))
    np.save(TRAIN_LABELS_FILE, np.array(train_labels, dtype=np.int32))
    np.save(TEST_PATHS_FILE, np.array(test_paths))
    np.save(TEST_LABELS_FILE, np.array(test_labels, dtype=np.int32))

    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f)

    print_success("Dataset preparation complete")
    return True


def _load_image(path, label):
    raw = tf.io.read_file(path)
    # A few Food-101 photos are cut off partway. Recover them when mostly complete.
    img = tf.io.decode_jpeg(
        raw, channels=3, try_recover_truncated=True, acceptable_fraction=0.8
    )
    img = tf.image.resize(img, IMAGE_SIZE)  # float32, values 0-255
    return img, label


def _skip_unreadable(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Safety net: silently drop any image that still cannot be decoded."""
    if hasattr(ds, "ignore_errors"):
        return ds.ignore_errors()
    return ds.apply(tf.data.experimental.ignore_errors())


def make_dataset(
    paths: np.ndarray, labels: np.ndarray, training: bool, num_classes: int = 0
) -> tf.data.Dataset:
    """Streams images from disk in parallel (no large memory cache)."""
    ds = tf.data.Dataset.from_tensor_slices(
        ([str(p) for p in paths], np.asarray(labels, dtype=np.int32))
    )
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    ds = _skip_unreadable(ds)
    if num_classes:
        ds = ds.map(
            lambda x, y: (x, tf.one_hot(y, num_classes)), num_parallel_calls=AUTOTUNE
        )
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def pil_to_model_input(image: Image.Image) -> np.ndarray:
    """Same preprocessing as training: fix rotation, convert to RGB, resize."""
    img = ImageOps.exif_transpose(image).convert("RGB")
    return tf.image.resize(np.array(img), IMAGE_SIZE).numpy().astype(np.float32)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(num_classes: int) -> Tuple[models.Model, models.Model]:
    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",  # only active during training
    )

    base = tf.keras.applications.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = base(x, training=False)  # EfficientNet rescales 0-255 inputs internally
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="plateiq"), base


def find_backbone(model: models.Model) -> models.Model:
    """Locate the pretrained EfficientNet inside a loaded model."""
    for layer in model.layers:
        if isinstance(layer, models.Model) and layer.name.startswith("efficientnet"):
            return layer
    raise ValueError("Pretrained backbone not found in the saved model.")


def compile_model(model: models.Model, lr) -> None:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=TOP_K, name="top5_accuracy"),
        ],
    )


def make_callbacks(patience: int, reduce_lr: bool) -> list:
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1
        ),
        # Saves the best model after every improvement, so progress is never lost
        callbacks.ModelCheckpoint(
            str(BEST_MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]
    if reduce_lr:
        cbs.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1
            )
        )
    return cbs


def train_model(resume: bool = False) -> bool:
    print_section("MODEL TRAINING")
    ensure_dirs()

    if not TRAIN_PATHS_FILE.exists() or not TRAIN_LABELS_FILE.exists():
        print_error("Training data not found. Run --prepare-data first.")
        return False

    classes = load_class_names()
    if not classes:
        print_error("Class list not found. Run --prepare-data first.")
        return False
    num_classes = len(classes)

    paths = np.load(TRAIN_PATHS_FILE)
    labels = np.load(TRAIN_LABELS_FILE)
    if len(paths) == 0:
        print_error("No training data available")
        return False

    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=VAL_SPLIT, stratify=labels, random_state=SEED
    )
    print_info(f"Food categories: {num_classes}")
    print_info(f"Training samples: {len(tr_p)}")
    print_info(f"Validation samples: {len(va_p)}")

    train_ds = make_dataset(tr_p, tr_l, training=True, num_classes=num_classes)
    val_ds = make_dataset(va_p, va_l, training=False, num_classes=num_classes)

    keys = ("accuracy", "val_accuracy", "loss", "val_loss")
    history = {k: [] for k in keys}
    finetune_start = 0

    try:
        if resume:
            if not BEST_MODEL_PATH.exists():
                print_error("No saved model to resume from. Run --train without --resume.")
                return False
            print_info("Resuming from the saved model. Skipping phase 1.")
            model = tf.keras.models.load_model(BEST_MODEL_PATH)
            base = find_backbone(model)
        else:
            model, base = build_model(num_classes)
            print_info("Phase 1: training the new classification layers...")
            compile_model(model, HEAD_LR)
            h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                           callbacks=make_callbacks(patience=2, reduce_lr=True), verbose=1)
            for k in keys:
                history[k] += [float(v) for v in h1.history[k]]
            finetune_start = len(history["accuracy"])

        print_info("Phase 2: fine-tuning the pretrained network...")
        base.trainable = True
        for layer in base.layers[:-UNFREEZE_LAST_LAYERS]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        total_steps = int(np.ceil(len(tr_p) / BATCH_SIZE)) * EPOCHS_FINE
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=FINE_TUNE_LR, decay_steps=total_steps, alpha=0.01
        )
        compile_model(model, schedule)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
                       callbacks=make_callbacks(patience=3, reduce_lr=False), verbose=1)
        for k in keys:
            history[k] += [float(v) for v in h2.history[k]]
    except KeyboardInterrupt:
        print_info("Training stopped. The best model so far is saved. Use --resume to continue.")
        return False
    except Exception as e:
        print_error(f"Training failed: {e}")
        return False

    if not BEST_MODEL_PATH.exists():
        model.save(BEST_MODEL_PATH)
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f)
    print_success(f"Model saved to {BEST_MODEL_PATH}")

    history["finetune_start"] = finetune_start
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    print_success(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    return True


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate_model() -> bool:
    print_section("MODEL EVALUATION")
    ensure_dirs()

    if not BEST_MODEL_PATH.exists():
        print_error("Model not found. Run --train first.")
        return False
    if not TEST_PATHS_FILE.exists() or not TEST_LABELS_FILE.exists():
        print_error("Test data not found. Run --prepare-data first.")
        return False

    classes = load_class_names()
    names = [display_name(c) for c in classes]
    num_classes = len(classes)

    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    test_paths = np.load(TEST_PATHS_FILE)
    test_labels = np.load(TEST_LABELS_FILE)
    print_info(f"Evaluating on {len(test_paths)} test images...")

    test_ds = make_dataset(test_paths, test_labels, training=False)
    steps = int(np.ceil(len(test_paths) / BATCH_SIZE))

    # Average the prediction for each image and its mirror image (small free boost)
    prob_batches, label_batches = [], []
    for x, y in tqdm(test_ds, total=steps, desc="Predicting"):
        p = np.asarray(model.predict_on_batch(x))
        p_flip = np.asarray(model.predict_on_batch(tf.image.flip_left_right(x)))
        prob_batches.append((p + p_flip) / 2)
        label_batches.append(y.numpy())
    probs = np.concatenate(prob_batches, axis=0)
    y_true = np.concatenate(label_batches, axis=0)

    skipped = len(test_paths) - len(y_true)
    if skipped > 0:
        print_info(f"{skipped} unreadable test images were skipped.")

    preds = np.argmax(probs, axis=1)
    top_k = np.argsort(probs, axis=1)[:, -TOP_K:]
    top5_accuracy = float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))
    accuracy = accuracy_score(y_true, preds)
    print_success(f"Top-1 Accuracy: {accuracy:.4f}")
    print_success(f"Top-5 Accuracy: {top5_accuracy:.4f}")

    label_ids = list(range(num_classes))
    report = classification_report(
        y_true, preds, labels=label_ids, target_names=names,
        output_dict=True, zero_division=0,
    )
    CLASS_REPORT_PATH.write_text(
        classification_report(
            y_true, preds, labels=label_ids, target_names=names, zero_division=0
        )
    )
    cm = confusion_matrix(y_true, preds, labels=label_ids)

    # Most frequently confused pairs of foods
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    flat_order = np.argsort(off_diag, axis=None)[::-1][:15]
    confused = []
    for idx in flat_order:
        a, b = np.unravel_index(idx, off_diag.shape)
        if off_diag[a, b] > 0:
            confused.append({"Actual": names[a], "Predicted as": names[b], "Count": int(off_diag[a, b])})

    per_class = [
        {
            "Food": n,
            "Precision": round(float(report[n]["precision"]), 4),
            "Recall": round(float(report[n]["recall"]), 4),
            "F1-score": round(float(report[n]["f1-score"]), 4),
        }
        for n in names
    ]

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "test_accuracy": float(accuracy),
                "top5_accuracy": top5_accuracy,
                "classification_report": report,
                "per_class": per_class,
                "most_confused": confused,
            },
            f, indent=2, default=float,
        )

    plot_training_history()
    plot_confusion_matrix(cm, names)
    plot_per_class_f1(per_class)
    print_success("Evaluation complete. Results saved in the 'results' folder.")
    return True


def plot_training_history() -> None:
    if not HISTORY_PATH.exists():
        return
    with open(HISTORY_PATH) as f:
        history = json.load(f)
    split = history.get("finetune_start")

    for key, title, path in (
        ("accuracy", "Model Accuracy", ACCURACY_PLOT_PATH),
        ("loss", "Model Loss", LOSS_PLOT_PATH),
    ):
        plt.figure(figsize=(7, 5))
        plt.plot(history[key], label="Training", marker="o")
        plt.plot(history[f"val_{key}"], label="Validation", marker="s")
        if split:
            plt.axvline(split - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(key.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_confusion_matrix(cm: np.ndarray, names: List[str]) -> None:
    row_sums = np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    normalised = cm / row_sums
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        normalised, cmap="Blues", xticklabels=names, yticklabels=names,
        square=True, cbar_kws={"label": "Share of each actual class"},
    )
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)
    plt.title("Confusion Matrix (normalised by actual class)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(per_class: List[Dict]) -> None:
    df = pd.DataFrame(per_class).sort_values("F1-score")
    fig, ax = plt.subplots(figsize=(9, 22))
    ax.barh(df["Food"], df["F1-score"], color="#2E6F95")
    ax.set_xlabel("F1-score")
    ax.set_title("F1-score by Food Category")
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(PER_CLASS_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict_pil(model: models.Model, image: Image.Image, classes: List[str]) -> Dict:
    x = pil_to_model_input(image)
    batch = np.stack([x, x[:, ::-1, :]])  # image + mirrored copy
    probs = np.asarray(model.predict(batch, verbose=0)).mean(axis=0)
    top = np.argsort(probs)[::-1][:TOP_K]
    return {
        "predicted_class": display_name(classes[top[0]]),
        "confidence": float(probs[top[0]]),
        "top_k": [
            {"class": display_name(classes[i]), "confidence": float(probs[i])} for i in top
        ],
    }


def predict_image(image_path: str, model_path: str = str(BEST_MODEL_PATH)) -> Dict:
    if not Path(model_path).exists():
        return {"error": "Model not found. Run --train first."}
    classes = load_class_names()
    if not classes:
        return {"error": "Class list not found."}
    try:
        image = Image.open(image_path)
    except Exception:
        return {"error": "Cannot load image."}
    model = tf.keras.models.load_model(model_path)
    return predict_pil(model, image, classes)


# ----------------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------------
def run_streamlit() -> None:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def load_model_cached(path: str, modified: float):
        return tf.keras.models.load_model(path)

    st.set_page_config(page_title="PlateIQ | Food Recognition", layout="wide")

    st.markdown(
        """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1F3A4D; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #5B6770; margin-bottom: 1.5rem; }
    .prediction-result { font-size: 1.6rem; font-weight: 600; color: #1F3A4D; padding: 1rem 1.25rem;
        background: #F2F6F9; border-left: 5px solid #2E6F95; border-radius: 4px; margin: 0.5rem 0 1rem 0; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">PlateIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Food recognition across 101 dishes using deep learning</div>',
        unsafe_allow_html=True,
    )

    classes = load_class_names()
    if not classes:
        st.error("Class list not found. Run `python plateiq.py --prepare-data` and `--train` first.")
        st.stop()
    names = [display_name(c) for c in classes]

    metrics = None
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    tab_overview, tab_predict, tab_foods, tab_model, tab_perf = st.tabs(
        ["Overview", "Recognise", "Supported Foods", "Model", "Performance"]
    )

    with tab_overview:
        st.header("About PlateIQ")
        st.write(
            "PlateIQ identifies food from photographs. It builds on EfficientNetB0, a network "
            "pretrained on millions of images, and is fine-tuned on the Food-101 dataset."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Food categories", len(names))
        if metrics:
            c2.metric("Top-1 accuracy", f"{metrics['test_accuracy']:.2%}")
            c3.metric("Top-5 accuracy", f"{metrics.get('top5_accuracy', 0):.2%}")

        st.subheader("How it works")
        st.write(
            f"""
        1. **Upload** a photo of a dish (JPG, JPEG or PNG).
        2. **Preprocess** - the image is straightened and resized to {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}.
        3. **Predict** - the model scores every one of the {len(names)} categories.
        4. **Review** - the top {TOP_K} matches are shown with confidence scores.
        """
        )

    with tab_predict:
        st.header("Food Recognition")
        uploaded_file = st.file_uploader(
            "Upload a food image", type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )

        if uploaded_file is not None:
            try:
                image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Uploaded image")
                    st.image(image)

                with col2:
                    st.subheader("Result")
                    if not BEST_MODEL_PATH.exists():
                        st.error("Model not found. Train it first with `python plateiq.py --train`.")
                    else:
                        with st.spinner("Analysing image..."):
                            model = load_model_cached(
                                str(BEST_MODEL_PATH), BEST_MODEL_PATH.stat().st_mtime
                            )
                            result = predict_pil(model, image, classes)

                        st.markdown(
                            f'<div class="prediction-result">{result["predicted_class"]}</div>',
                            unsafe_allow_html=True,
                        )
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                        if result["confidence"] < CONFIDENCE_THRESHOLD:
                            st.warning(
                                "Low confidence. The image may not show one of the "
                                f"{len(names)} foods PlateIQ was trained on."
                            )

                        st.subheader(f"Top {TOP_K} matches")
                        for i, pred in enumerate(result["top_k"]):
                            col_a, col_b = st.columns([3, 2])
                            with col_a:
                                st.write(f"{i + 1}. {pred['class']}")
                            with col_b:
                                st.progress(min(max(pred["confidence"], 0.0), 1.0))
                                st.caption(f"{pred['confidence']:.2%}")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    with tab_foods:
        st.header("Supported Foods")
        query = st.text_input("Search", placeholder="For example: pizza")
        matches = [n for n in names if query.strip().lower() in n.lower()]
        st.caption(f"Showing {len(matches)} of {len(names)} categories")
        cols = st.columns(4)
        for i, name in enumerate(matches):
            cols[i % 4].write(name)

    with tab_model:
        st.header("Model Architecture")
        arch = pd.DataFrame(
            {
                "Layer": [
                    "Input", "Data augmentation", "EfficientNetB0 (pretrained)",
                    "Global average pooling", "Dropout (0.3)", f"Output ({len(names)}, softmax)",
                ],
                "Description": [
                    f"RGB image, {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}",
                    "Random flip, rotation, zoom, shift and contrast (training only)",
                    "Feature extractor trained on ImageNet and fine-tuned on Food-101",
                    "Summarises image features into one vector",
                    "Reduces overfitting",
                    "Probability for each food category",
                ],
            }
        )
        st.table(arch)

        st.subheader("Training configuration")
        config = pd.DataFrame(
            {
                "Parameter": [
                    "Optimizer", "Loss", "Phase 1", "Phase 2", "Batch size", "Early stopping",
                ],
                "Value": [
                    "Adam",
                    f"Categorical cross-entropy (label smoothing {LABEL_SMOOTHING})",
                    f"Up to {EPOCHS_HEAD} epochs, learning rate {HEAD_LR:g} (new layers only)",
                    f"Up to {EPOCHS_FINE} epochs, learning rate decaying from {FINE_TUNE_LR:g} "
                    f"(top {UNFREEZE_LAST_LAYERS} layers unfrozen)",
                    str(BATCH_SIZE),
                    "Enabled, best model is saved",
                ],
            }
        )
        st.table(config)

    with tab_perf:
        st.header("Model Performance")
        if metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top-1 accuracy", f"{metrics['test_accuracy']:.2%}")
            c2.metric("Top-5 accuracy", f"{metrics.get('top5_accuracy', 0):.2%}")
            c3.metric("Macro F1-score", f"{metrics['classification_report']['macro avg']['f1-score']:.2%}")
            c4.metric("Weighted F1-score", f"{metrics['classification_report']['weighted avg']['f1-score']:.2%}")

            st.subheader("Training history")
            h1, h2 = st.columns(2)
            with h1:
                if ACCURACY_PLOT_PATH.exists():
                    st.image(str(ACCURACY_PLOT_PATH))
            with h2:
                if LOSS_PLOT_PATH.exists():
                    st.image(str(LOSS_PLOT_PATH))

            per_class = pd.DataFrame(metrics.get("per_class", []))
            if not per_class.empty:
                st.subheader("Strongest and weakest categories")
                s1, s2 = st.columns(2)
                with s1:
                    st.caption("Highest F1-score")
                    st.dataframe(per_class.sort_values("F1-score", ascending=False).head(10), hide_index=True)
                with s2:
                    st.caption("Lowest F1-score")
                    st.dataframe(per_class.sort_values("F1-score").head(10), hide_index=True)

            confused = pd.DataFrame(metrics.get("most_confused", []))
            if not confused.empty:
                st.subheader("Most frequently confused foods")
                st.dataframe(confused, hide_index=True)

            if CONFUSION_MATRIX_PATH.exists():
                st.subheader("Confusion matrix")
                st.image(str(CONFUSION_MATRIX_PATH))

            if PER_CLASS_PLOT_PATH.exists():
                st.subheader("F1-score by category")
                st.image(str(PER_CLASS_PLOT_PATH))

            if CLASS_REPORT_PATH.exists():
                with st.expander("Full classification report"):
                    st.text(CLASS_REPORT_PATH.read_text())
        else:
            st.warning("No evaluation results found. Run `python plateiq.py --evaluate` first.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    global SUBSET_SIZE, EPOCHS_HEAD, EPOCHS_FINE, BATCH_SIZE

    if len(sys.argv) == 1:
        run_streamlit()
        return

    parser = argparse.ArgumentParser(description="PlateIQ - Food Recognition (101 categories)")
    parser.add_argument("--prepare-data", action="store_true", help="Download and prepare the dataset")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--resume", action="store_true",
                        help="With --train: continue from the saved model (skips phase 1)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model")
    parser.add_argument("--predict", type=str, default=None, help="Predict the food in an image file")
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE,
                        help="Training images per class (max 750; lower it for a quicker run)")
    parser.add_argument("--epochs", type=int, default=EPOCHS_HEAD, help="Epochs for phase 1")
    parser.add_argument("--finetune-epochs", type=int, default=EPOCHS_FINE, help="Epochs for phase 2")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    SUBSET_SIZE = args.subset_size
    EPOCHS_HEAD = args.epochs
    EPOCHS_FINE = args.finetune_epochs
    BATCH_SIZE = args.batch_size

    print_section("PLATEIQ - FOOD RECOGNITION")
    check_gpu()
    verify_imports()

    if args.prepare_data:
        prepare_dataset(SUBSET_SIZE)
    if args.train:
        train_model(resume=args.resume)
    if args.evaluate:
        evaluate_model()
    if args.predict:
        result = predict_image(args.predict)
        if "error" in result:
            print_error(result["error"])
        else:
            print_success(f"Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
            for i, item in enumerate(result["top_k"], start=1):
                print(f"  {i}. {item['class']}: {item['confidence']:.2%}")


if __name__ == "__main__":
    main()
