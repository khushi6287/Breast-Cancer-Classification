# Deep Learning PR1 – Breast Cancer Classification

## Project Overview

This project implements a deep learning classification system using the **Breast Cancer Wisconsin (Diagnostic)** dataset.

The project explores different neural network architectures, activation functions, and regularization techniques using **TensorFlow and Keras**.

The main objective is to understand how different deep learning techniques affect model training, overfitting, and generalization.

---

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset available through `scikit-learn`.

### Dataset Details

- **Samples:** 569
- **Features:** 30 numeric features
- **Target Classes:** 2
- **Target 0:** Malignant
- **Target 1:** Benign
- **Missing Values:** None

The dataset is loaded using:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
```

---

## Technologies Used

- Python 3.x
- TensorFlow 2.x
- Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Google Colab / Jupyter Notebook

---

## Project Workflow

The project follows these major steps:

```text
Dataset Loading
      ↓
Exploratory Data Analysis
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Single-Layer Perceptron
      ↓
Multi-Layer Perceptron
      ↓
Activation Function Comparison
      ↓
Early Stopping
      ↓
Dropout Regularization
      ↓
L1 Regularization
      ↓
L2 Regularization
      ↓
L1-L2 Regularization
      ↓
Final Combined Model
      ↓
Model Evaluation
```

---

## Data Preprocessing

### Train-Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

The split uses:

```python
test_size=0.2
random_state=42
stratify=y
```

### Feature Scaling

`StandardScaler` is used to scale the features.

The scaler is fitted only on the training data and then applied to both training and testing data.

This helps the gradient-based optimizer train more consistently because the features have similar scales.

---

# Models Implemented

## 1. Single-Layer Perceptron

The SLP contains:

```text
30 input features
        ↓
1 Sigmoid neuron
```

The model uses:

- Adam optimizer
- Binary Cross-Entropy loss
- Accuracy metric
- 50 epochs
- Batch size of 32

The SLP provides a baseline model and can learn a linear decision boundary.

---

## 2. Multi-Layer Perceptron – ReLU

The MLP architecture is:

```text
30 → 64 → 32 → 1
```

Hidden layers use **ReLU** activation and the output layer uses **Sigmoid**.

The model is trained for 100 epochs.

---

## 3. Activation Function Comparison

Three activation functions are compared:

- ReLU
- Tanh
- Sigmoid

The validation accuracy and test accuracy are compared to understand the effect of different activation functions.

---

## 4. Early Stopping

A deeper MLP is created with:

```text
30 → 128 → 64 → 1
```

Early Stopping monitors:

```text
val_loss
```

with:

```text
patience = 15
restore_best_weights = True
```

The model is allowed to train for a maximum of 300 epochs.

A second model without Early Stopping is also trained for comparison.

---

## 5. Dropout Regularization

Dropout is applied to the hidden layers.

The main Dropout model uses:

```text
Dropout Rate = 0.3
```

Different dropout rates are also compared:

```text
0.1
0.3
0.5
```

The models are evaluated using validation accuracy and test accuracy.

---

## 6. L1 Regularization

L1 regularization is applied with:

```text
L1 = 0.001
```

L1 regularization can encourage some weights to become close to zero and can produce a sparse model.

---

## 7. L2 Regularization

L2 regularization is applied with:

```text
L2 = 0.001
```

L2 regularization encourages smaller weights and can help reduce overfitting.

---

## 8. L1-L2 Regularization

L1-L2 regularization combines both techniques.

The project uses:

```text
L1 = 0.0001
L2 = 0.001
```

This combines the sparsity effect of L1 with the weight-shrinkage effect of L2.

---

# Final Combined Model

The final model combines:

- ReLU activation
- L2 regularization
- Dropout
- Early Stopping

Architecture:

```text
30
 ↓
128 ReLU + L2
 ↓
Dropout 0.3
 ↓
64 ReLU + L2
 ↓
Dropout 0.3
 ↓
1 Sigmoid
```

The final model uses:

```text
L2 = 0.001
Dropout = 0.3
Early Stopping patience = 20
```

---

# Model Evaluation

The models are evaluated using:

### Accuracy

Measures the overall percentage of correctly classified observations.

### Precision

Measures how many predicted positive observations are actually positive.

### Recall

Measures how many actual positive observations are correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

### Confusion Matrix

The confusion matrix is used to visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

# Results

The notebook contains a comparison table with:

| Model | Architecture | Regularization | Dropout | Early Stopping | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|
| SLP | 30 → 1 | None | 0 | No | Generated in Notebook | Generated in Notebook | Generated in Notebook | Generated in Notebook |
| MLP-ReLU | 30 → 64 → 32 → 1 | None | 0 | No | Generated in Notebook | Generated in Notebook | Generated in Notebook | Generated in Notebook |
| MLP+ES | 30 → 128 → 64 → 1 | None | 0 | Yes | Generated in Notebook | Generated in Notebook | Generated in Notebook | Generated in Notebook |
| MLP+Dropout-best | 30 → 128 → 64 → 1 | Dropout | Best Rate | Yes | Generated in Notebook | Generated in Notebook | Generated in Notebook | Generated in Notebook |
| MLP+L2 | 30 → 128 → 64 → 1 | L2 | 0 | Yes | Generated in Notebook | Generated in Notebook | Generated in Notebook | Generated in Notebook |
| Final Combined Model | 30 → 128 → 64 → 1 | L2 | 0.3 | Yes | Generated in Notebook | Generated in Notebook | Generated in Notebook | Generated in Notebook |

> **Note:** The final values are generated automatically when the notebook is executed. Results may vary slightly between runs because neural-network training can involve random initialization.

---

# Project Visualizations

The notebook includes:

- Target Class Distribution
- Feature Correlation Heatmap
- SLP Training vs Validation Loss
- SLP Training vs Validation Accuracy
- Activation Function Comparison
- Early Stopping Loss Curve
- Validation Loss With vs Without Early Stopping
- Dropout Rate Comparison
- L1/L2/L1-L2 Loss Curves
- Regularization Test Accuracy Comparison
- Final Model Loss and Accuracy
- Final Model Confusion Matrix

---

# Clinical Insight

For this educational project, Precision and Recall are both important evaluation measures.

The classification threshold used in the project is:

```text
0.5
```

Changing the threshold can change the balance between precision and recall.

The regularization techniques and Early Stopping are used to reduce overfitting and improve generalization to unseen test data.

This project uses a public dataset for educational purposes. The trained model is **not intended to be used as a clinical diagnostic system or for real medical decisions**.

---

# Project Structure

Recommended GitHub repository structure:

```text
Deep-Learning-PR1/
│
├── DL_PR1.ipynb
├── DL_PR1.html
├── README.md
├── requirements.txt
│
├── plots/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── slp_loss.png
│   ├── slp_accuracy.png
│   ├── activation_comparison.png
│   ├── early_stopping.png
│   ├── dropout_comparison.png
│   ├── regularization_loss.png
│   └── confusion_matrix.png
│
└── video/
    └── project_demo_link.txt
```

---

# How to Run the Project

## Option 1: Google Colab

1. Open Google Colab.
2. Upload `DL_PR1.ipynb`.
3. Select **Runtime → Run all**.
4. Wait for all cells to execute.
5. Check the generated graphs and results.

## Option 2: Local Jupyter Notebook

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then open:

```bash
jupyter notebook
```

Open:

```text
DL_PR1.ipynb
```

and run all cells.

---

# Requirements

The main libraries required are:

```text
TensorFlow >= 2.12
scikit-learn >= 1.4
pandas
numpy
matplotlib
seaborn
```

---

# Learning Outcomes

After completing this project, the following concepts are demonstrated:

- Single-Layer Perceptron
- Multi-Layer Perceptron
- Feature Scaling
- ReLU activation
- Tanh activation
- Sigmoid activation
- Early Stopping
- Dropout
- L1 Regularization
- L2 Regularization
- L1-L2 Regularization
- Model Evaluation
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Overfitting and Generalization

---

# Conclusion

This project demonstrates how different neural-network architectures and regularization techniques can be applied to a binary classification problem.

The experiments compare SLP and MLP models and investigate the effects of activation functions, Early Stopping, Dropout, L1, L2 and L1-L2 regularization.

The final combined model integrates **L2 regularization, Dropout and Early Stopping** and is evaluated using Accuracy, Precision, Recall and F1-Score.

---


Deep Learning PR1

Red & White Skill Education
