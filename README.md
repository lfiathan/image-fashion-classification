# Dataset configuration
DATASET_SLUG = 'nguyngiabol/colorful-fashion-dataset-for-object-detection'
OUTPUT_PATH = './dataset'
DATASET_ROOT = os.path.join(OUTPUT_PATH, 'colorful_fashion_dataset_for_object_detection')
IMAGES_PATH = os.path.join(DATASET_ROOT, 'JPEGImages')
ORIGINAL_ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, 'Annotations_txt')

# Phase 1 classes
PHASE1_CLASSES = {'TOP': 0, 'BOTTOM': 1, 'SHOES': 2}
PHASE1_MAP = {
    0: None,  # sunglass
    1: None,  # hat
    2: 0,     # jacket -> TOP
    3: 0,     # shirt  -> TOP
    4: 1,     # pants  -> BOTTOM
    5: 1,     # shorts -> BOTTOM
    6: 1,     # skirt  -> BOTTOM
    7: 0,     # dress  -> TOP
    8: None,  # bag
    9: 2      # shoe   -> SHOES
}

# Phase 2 classes
PHASE2_CLASSES = {
    'jacket': 0, 'shirt': 1, 'pants': 2, 'shorts': 3,
    'skirt': 4, 'dress': 5, 'shoe': 6
}
PHASE2_MAP = {
    0: None,  # sunglass
    1: None,  # hat
    2: 0,     # jacket
    3: 1,     # shirt
    4: 2,     # pants
    5: 3,     # shorts
    6: 4,     # skirt
    7: 5,     # dress
    8: None,  # bag
    9: 6      # shoe
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'imgsz': 512,
    'batch': 16,
    'workers': 8,
    'patience': 10,
    'cache': True,
    'half': True  # Use mixed precision if supported
}
```

2. **dataset.py**: This file will handle dataset downloading and processing.

```python
import os
import kaggle
import shutil
from config import DATASET_SLUG, OUTPUT_PATH, DATASET_ROOT

def download_dataset():
    if not os.path.exists(OUTPUT_PATH):
        print(f"Dataset not found. Downloading '{DATASET_SLUG}' from Kaggle...")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        kaggle.api.dataset_download_files(DATASET_SLUG, path=OUTPUT_PATH, unzip=True)
        print("✅ Download and extraction complete.")
    else:
        print(f"✅ Dataset already found at '{OUTPUT_PATH}'. Skipping download.")

def create_label_directories():
    labels_phase1_dir = os.path.join(DATASET_ROOT, 'labels_phase1')
    labels_phase2_dir = os.path.join(DATASET_ROOT, 'labels_phase2')

    if os.path.exists(labels_phase1_dir):
        shutil.rmtree(labels_phase1_dir)
    os.makedirs(labels_phase1_dir)

    if os.path.exists(labels_phase2_dir):
        shutil.rmtree(labels_phase2_dir)
    os.makedirs(labels_phase2_dir)

    return labels_phase1_dir, labels_phase2_dir
```

3. **train.py**: This file will contain the training logic for both phases.

```python
import os
from ultralytics import YOLO
from config import TRAINING_CONFIG, DATASET_ROOT
from dataset import create_label_directories

def train_model(model_path, data_yaml, phase_name):
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        epochs=TRAINING_CONFIG['epochs'],
        imgsz=TRAINING_CONFIG['imgsz'],
        batch=TRAINING_CONFIG['batch'],
        workers=TRAINING_CONFIG['workers'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache=TRAINING_CONFIG['cache'],
        half=TRAINING_CONFIG['half'],
        patience=TRAINING_CONFIG['patience'],
        name=phase_name,
        exist_ok=True
    )
    print(f"--- ✅ {phase_name} TRAINING COMPLETE ---")

def main():
    download_dataset()
    labels_phase1_dir, labels_phase2_dir = create_label_directories()

    # Phase 1 training
    print("\n--- 🚀 STARTING PHASE 1 TRAINING ---")
    os.rename(labels_phase1_dir, os.path.join(DATASET_ROOT, 'labels'))
    train_model('yolov8n.pt', 'phase1-data.yaml', 'yolov8_phase1_coarse')

    # Phase 2 training
    print("\n--- 🚀 STARTING PHASE 2 TRAINING ---")
    os.rename(labels_phase2_dir, os.path.join(DATASET_ROOT, 'labels'))
    train_model('runs/detect/yolov8_phase1_coarse/weights/best.pt', 'phase2-data.yaml', 'yolov8_phase2_fine')

if __name__ == "__main__":
    main()
```

4. **predict.py**: This file will handle the prediction logic.

```python
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from config import IMAGES_PATH

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def run_prediction():
    model_coarse = YOLO('runs/detect/yolov8_phase1_coarse/weights/best.pt')
    model_fine = YOLO('runs/detect/yolov8_phase2_fine/weights/best.pt')

    test_image_name = random.choice(os.listdir(IMAGES_PATH))
    test_image_path = os.path.join(IMAGES_PATH, test_image_name)
    print(f"-> Testing on image: {test_image_name}")

    img = cv2.imread(test_image_path)
    results_coarse = model_coarse(img, verbose=False)[0]
    results_fine = model_fine(img, verbose=False)[0]
    IOU_THRESHOLD = 0.7

    for fine_box in results_fine.boxes:
        fine_xyxy = fine_box.xyxy[0].cpu().numpy()
        fine_class_name = model_fine.names[int(fine_box.cls)]
        best_match_coarse_name = "Unknown"
        max_iou = 0

        for coarse_box in results_coarse.boxes:
            iou = calculate_iou(fine_xyxy, coarse_box.xyxy[0].cpu().numpy())
            if iou > max_iou:
                max_iou = iou
                best_match_coarse_name = model_coarse.names[int(coarse_box.cls)]

        label = f"{best_match_coarse_name}: {fine_class_name}" if max_iou > IOU_THRESHOLD else fine_class_name
        x1, y1, x2, y2 = map(int, fine_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Hierarchical Prediction Result")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_prediction()
```

5. **features.py**: This file can be used for any additional feature engineering or processing functions. For now, it will be left intentionally blank.

```python
# This file is intentionally left blank.
```

6. **__init__.py**: This file will be left intentionally blank as well.

```python
# This file is intentionally left blank.
```

With these files structured, you can now run the training and prediction processes more efficiently and make adjustments to hyperparameters in the `config.py` file as needed.# image-fashion-classification
