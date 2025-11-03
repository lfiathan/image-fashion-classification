import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

ALLOWED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def gather_images(dirpath: Path):
    images = []
    labels = []
    classes = sorted([p.name for p in dirpath.iterdir() if p.is_dir()])
    for cls in classes:
        for p in (dirpath / cls).iterdir():
            if p.is_file() and p.suffix.lower() in ALLOWED:
                images.append(str(p))
                labels.append(cls)
    return images, labels, classes

def _probs_to_array(probs):
    # robust conversion for various Probs-like objects
    try:
        if hasattr(probs, "cpu") and hasattr(probs, "numpy"):
            return probs.cpu().numpy().ravel()
        if hasattr(probs, "numpy"):
            return probs.numpy().ravel()
        if isinstance(probs, (list, tuple, np.ndarray)):
            return np.asarray(probs).ravel()
        # try iterating
        return np.asarray(list(probs)).ravel()
    except Exception:
        return np.asarray([float(x) for x in probs]).ravel()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", "-w", default="runs-cls/yolov8n-cls-fashion/weights/best.pt")
    p.add_argument("--dir", "-d", required=True, help="Folder with class subfolders (train/ or val/)")
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    model = YOLO(args.weights)
    img_dir = Path(args.dir)
    if not img_dir.exists():
        print("Dir not found:", img_dir)
        sys.exit(1)

    imgs, true_labels, classes = gather_images(img_dir)
    if not imgs:
        print("No images found under", img_dir)
        sys.exit(1)

    # Build model_classes and mapping used for true labels
    model_classes = None
    try:
        # model.names often maps int->str
        name_map = {int(k): v for k, v in model.names.items()}
        model_classes = [name_map[i] for i in sorted(name_map.keys())]
    except Exception:
        model_classes = classes

    class_to_idx = {n: i for i, n in enumerate(model_classes)}
    true_idx = [class_to_idx.get(lbl, -1) for lbl in true_labels]
    if any(t == -1 for t in true_idx):
        print("Warning: some true labels not found in model/classes mapping")

    print(f"Running {len(imgs)} predictions (batch={args.batch})...")
    # use stream=True to avoid accumulating all Results in memory
    results_stream = model.predict(source=imgs, imgsz=args.imgsz, batch=args.batch, verbose=False, stream=True)

    pred_idx = []
    for r in results_stream:
        probs = getattr(r, "probs", None)
        if probs is None:
            raise RuntimeError("Results object has no 'probs' attribute — make sure this is a classification model")

        # robust conversion of various Probs-like types to a numeric array
        try:
            if hasattr(probs, "cpu") and hasattr(probs, "numpy"):
                arr = probs.cpu().numpy().ravel()
            elif hasattr(probs, "numpy"):
                arr = probs.numpy().ravel()
            elif isinstance(probs, (list, tuple, np.ndarray)):
                arr = np.asarray(probs).ravel()
            else:
                # fall back to iterator -> list -> numpy
                arr = np.asarray(list(probs)).ravel()
        except Exception:
            # last resort: try converting each element to float
            arr = np.array([float(x) for x in probs]).ravel()

        if arr.size == 0:
            raise RuntimeError("Empty probability array for a result")
        pred_idx.append(int(arr.argmax()))

    # Align lengths
    n = min(len(true_idx), len(pred_idx))
    true_idx = np.array(true_idx[:n])
    pred_idx = np.array(pred_idx[:n])

    print("\nClassification report (classes shown by index -> name):")
    for i, name in enumerate(model_classes):
        print(f" {i:2d} -> {name}")

    print("\n" + classification_report(true_idx, pred_idx, target_names=[str(n) for n in model_classes], digits=4))
    print("F1 (macro):", f1_score(true_idx, pred_idx, average='macro'))
    print("F1 (micro):", f1_score(true_idx, pred_idx, average='micro'))
    print("Precision (macro):", precision_score(true_idx, pred_idx, average='macro'))
    print("Recall (macro):", recall_score(true_idx, pred_idx, average='macro'))

if __name__ == "__main__":
    main()