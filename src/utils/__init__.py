import os

# Dataset paths
DATASET_ROOT = './dataset/colorful_fashion_dataset_for_object_detection'
IMAGES_PATH = os.path.join(DATASET_ROOT, 'JPEGImages')
ORIGINAL_ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, 'Annotations_txt')

# Label directories
LABELS_PHASE1_DIR = os.path.join(DATASET_ROOT, 'labels_phase1')
LABELS_PHASE2_DIR = os.path.join(DATASET_ROOT, 'labels_phase2')

# Training parameters
TRAINING_PARAMS = {
    'epochs': 50,
    'imgsz': 512,
    'batch': 16,
    'workers': 8,
    'patience': 10,
    'cache': True,
}