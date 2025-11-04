# YOLOv8 Fashion Image Classification

This project details the process of training a high-accuracy, 16-class fashion item classifier using YOLOv8-cls (from `ultralytics`). The primary challenge was sourcing, cleaning, and balancing data from multiple disparate datasets.

The final model achieves **~97% top-1 accuracy** on the validation set.

### Features

The model is trained to classify 16 distinct fashion categories:
- blazer
- boot
- dress
- flip-flop
- hoodie
- jacket
- loafer
- pants
- polo
- shirt
- short
- skirt
- slipper
- sneaker
- sweater
- t-shirt

### How It Works: A Step-by-Step Tutorial

This project was built and documented in the `notebooks/modisch-model-classification.ipynb` notebook. Here is a summary of the development pipeline:

**1. Data Sourcing & Download**
* The project uses the Kaggle API to download two separate datasets:
    * `ryanbadai/clothes-dataset`
    * `noobyogi0100/shoe-dataset`
* The notebook handles the automatic download and unzipping of these files into a local `data/` directory.

**2. Data Preparation & Re-mapping**
* The two source datasets use different class names and languages (e.g., "Celana_Panjang", "Kaos", "sneakers").
* A key step was creating a mapping dictionary (`CLOTHES_MAP` and `SHOES_MAP`) to standardize these source labels into the 16 target classes.
* Files were copied from their source folders into a new `dataset-fashion-modisch/` directory, structured for YOLO training (`train/` and `val/`) with an 80/20 split.

**3. Data Balancing (Oversampling)**
* Initial analysis showed a severe class imbalance. For example, the 'jacket' class had 1995 training images, while 'boot' and 'slipper' had only 249.
* To fix this, the notebook performs oversampling *only on the training set*. It iteratively copies and augments images from minority classes (using PIL for random mirroring, rotation, and color jitter) until all 16 classes have the same number of samples as the majority class (1995 images).

**4. Model Training**
* A `yolov8n-cls` (YOLOv8-Nano Classification) model from `ultralytics` was used.
* The model was trained for 30 epochs with a batch size of 64 and an image size of 224x224.
* The final `best.pt` model weights, achieving ~97% top-1 accuracy, are saved in the `runs-cls/` directory.

**5. Model Evaluation & Export**
* The model was validated against the test set to confirm its performance.
* The `testing_and_visualize.ipynb` notebook was used to generate detailed confusion matrices (both raw counts and normalized) to see exactly where the model was succeeding or failing by class.
* An F1-score vs. Confidence Threshold analysis was also performed to determine an optimal threshold for real-world predictions.
* Finally, the trained PyTorch (`.pt`) model was exported to ONNX (`.onnx`) format for deployment-ready inference.

---

### Getting Started (How to Fork & Run)

Follow these steps to replicate the project and re-train the model.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/image-fashion-classification.git](https://github.com/your-username/image-fashion-classification.git)
cd image-fashion-classification
```

**2. Set Up Environment**

  * It is highly recommended to use a Python virtual environment.
  * Install all required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

**3. Set Up Kaggle API**

  * This project requires the Kaggle API to download data.
  * Follow the official Kaggle instructions to create an API token (`kaggle.json`).
  * Place the `kaggle.json` file in the required directory (e.g., `~/.kaggle/kaggle.json`).

**4. Run the Notebook**

  * Open and run the main notebook:
    `notebooks/modisch-model-classification.ipynb`
  * The notebook will automatically:
    1.  Download and unzip the datasets.
    2.  Process, map, and balance the data.
    3.  Train the YOLOv8-cls model.
    4.  Save the final weights and ONNX file.

**5. (Optional) Run Visualization**

  * To see the confusion matrix and other plots, run:
    `notebooks/testing_and_visualize.ipynb`

### Dataset Credits

This model would not be possible without the public datasets provided by the community.

  * **Clothes Dataset:** [https://www.kaggle.com/datasets/ryanbadai/clothes-dataset](https://www.kaggle.com/datasets/ryanbadai/clothes-dataset)
  * **Shoe Dataset:** [https://www.kaggle.com/datasets/noobyogi0100/shoe-dataset](https://www.kaggle.com/datasets/noobyogi0100/shoe-dataset)
