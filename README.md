# 🩺 Chest X-ray Pneumonia Detection using CNN

## 🧠 Overview
This project uses **Convolutional Neural Networks (CNN)** to detect **Pneumonia** from **Chest X-ray images**.  
By leveraging deep learning, the model classifies X-ray images as **Normal** or **Pneumonia**, providing an efficient tool to assist medical diagnosis — especially in areas with limited radiology resources.

---

## 📑 Table of Contents
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)
- [Folder Structure](#folder-structure)
- [Future Improvements](#future-improvements)
- [References](#references)
- [License](#license)

---

## 💡 Motivation
- Pneumonia remains one of the leading causes of death worldwide, particularly among children and the elderly.  
- Detecting pneumonia through X-rays requires expert radiologists, which are not always accessible.  
- This project aims to automate pneumonia detection using CNNs, enabling faster and more consistent diagnostics.

---
## 📂 Dataset
- **Source:** [Chest X-Ray Images (Pneumonia) - Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **Structure:**
```bash
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```
- **Total Images:**
  - Train: 5,216 (Normal: 1,341, Pneumonia: 3,875)
  - Validation: 16 (Normal: 8, Pneumonia: 8)
  - Test: 624 (Normal: 234, Pneumonia: 390)

- **Preprocessing:**
  - Images resized to 150×150 pixels  
  - Pixel normalization (rescale = 1/255)  
  - Data augmentation (rotation, flip, zoom)


---

## ⚙️ Approach
1. **Data Preprocessing:** Using Keras `ImageDataGenerator` for augmentation and normalization.  
2. **Model Design:** Built a CNN from scratch using TensorFlow/Keras.  
3. **Training:** Tuned hyperparameters for optimal accuracy without overfitting.  
4. **Evaluation:** Used metrics such as accuracy, precision, recall, and F1-score.  
5. **Visualization:** Plotted accuracy/loss curves and confusion matrix for analysis.

---

## 🧩 Model Architecture

| Layer | Type | Output Shape | Parameters | Activation |
|-------|------|--------------|-----------|-----------|
| 1 | Conv2D | (None, 148, 148, 32) | 896 | ReLU |
| 2 | MaxPooling2D | (None, 74, 74, 32) | 0 | - |
| 3 | BatchNormalization | (None, 74, 74, 32) | 128 | - |
| 4 | Conv2D | (None, 72, 72, 64) | 18,496 | ReLU |
| 5 | MaxPooling2D | (None, 36, 36, 64) | 0 | - |
| 6 | BatchNormalization | (None, 36, 36, 64) | 256 | - |
| 7 | Conv2D | (None, 34, 34, 128) | 73,856 | ReLU |
| 8 | MaxPooling2D | (None, 17, 17, 128) | 0 | - |
| 9 | BatchNormalization | (None, 17, 17, 128) | 512 | - |
| 10 | Dropout | (None, 17, 17, 128) | 0 | - |
| 11 | Flatten | (None, 36992) | 0 | - |
| 12 | Dense | (None, 128) | 4,735,104 | ReLU |
| 13 | Dropout | (None, 128) | 0 | - |
| 14 | Dense | (None, 1) | 129 | Sigmoid |

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Learning Rate:** 0.001  
**Epochs:** 20  
**Batch Size:** 32  
**Total Parameters:** 4,829,377  
**Trainable Parameters:** 4,828,929  
**Non-Trainable Parameters:** 448

---

## 📈 Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC Curve (AUC)**

---

## 🧰 Requirements

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```
Example requirements.txt:
```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
```

## ▶️ How to Run

1. **Clone the repository**
```
git clone https://github.com/SourabhKhamankar22/Chest-xray-Pneumonia-detection-using-CNN.git
cd Chest-xray-Pneumonia-detection-using-CNN
```
2. **Place the dataset:**
 Download the dataset from Kaggle and extract it into the data/ folder as shown above.

3. **Run the notebook**
```
jupyter notebook pneumonia_detection.ipynb
```
   or open it directly in Google Colab.

4. **Steps performed in the notebook:**

  - **Load and preprocess dataset**
  - **Build CNN model**
  - **Train and validate**
  - **Evaluate performance**
  - **Visualize results**

## 📊 Results

- Achieved **93% validation accuracy** and **89% test accuracy**
- Model generalizes well on unseen test images
- **Classification Report on Test Set:**
  - **NORMAL:** Precision 0.92 | Recall 0.74 | F1-score 0.82 | Support 234  
  - **PNEUMONIA:** Precision 0.86 | Recall 0.96 | F1-score 0.91 | Support 390  
- **Overall Metrics:** Accuracy 0.88 | Precision 0.88 | Recall 0.90 | F1-score 0.89
- **Confusion Matrix & ROC Curve** demonstrate strong discriminative ability
- Training and validation curves show stable convergence with minimal overfitting

 
## 🖼 Visualizations
- **Accuracy & Loss Curves:**  
  ![Accuracy & Loss Curves](images/accuracy_loss.png)
- **Confusion Matrix:**  
  ![Confusion Matrix](images/confusion_matrix.png)
- **Sample predictions on test images (Normal vs Pneumonia)**  
  ![Sample Predictions](images/sample_prediction.png)

## 🗂 Folder Structure
```
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── pneumonia_model.h5
├── pneumonia_detection.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Future Improvements

  - **Implement Transfer Learning (e.g., ResNet50, DenseNet, EfficientNet)**
  - **Use cross-validation to improve robustness**
  - **Add Grad-CAM visualization for model interpretability**
  - **Deploy model as a web app (using Streamlit or Flask)**
  - **Train with larger and more diverse datasets**

## 📚 References

  - **Kaggle: Chest X-Ray Images (Pneumonia)**
  - **TensorFlow & Keras Documentation**
  - **Related research on Pneumonia detection using deep learning**

## 📝 License

  This project is released under the MIT License

## Author: [Sourabh Khamankar](https://github.com/SourabhKhamankar22)

  🎯 Deep Learning | Medical Imaging | CNN | TensorFlow
