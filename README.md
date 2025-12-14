# Breast Cancer Diagnosis Classification 🧬

This project implements **end-to-end machine learning pipelines** for **breast cancer diagnosis (Benign vs Malignant)** using the **Wisconsin Breast Cancer Dataset**.

The work combines:

* **Data analysis & visualization**
* **ML algorithms implemented from scratch**
* **Classical ML models using scikit-learn**
* **Dimensionality reduction with PCA**
* **Model evaluation using multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC)**

The final system identifies **SVM (RBF kernel)** as the best-performing model.

---

## 📂 Dataset

* **Source**: UCI Machine Learning Repository (via Kaggle)
* **Dataset**: Breast Cancer Wisconsin (Diagnostic)
* **Target Variable**:

  * `0` → Benign
  * `1` → Malignant

The dataset contains **30 real-valued features** computed from digitized images of breast mass cell nuclei.

---

## ⚙️ Project Workflow

1. **Data Loading**

   * Downloaded directly using `kagglehub`
   * Loaded into pandas DataFrame

2. **Data Cleaning**

   * Removed non-informative columns (`id`, `Unnamed: 32`)
   * Converted diagnosis labels from `M/B` → `1/0`

3. **Exploratory Data Analysis (EDA)**

   * Class distribution visualization
   * Feature-wise histograms
   * Correlation heatmap

4. **Train–Test Split (From Scratch)**

   * Custom implementation using NumPy
   * 80% training, 20% testing

5. **Evaluation Metrics (From Scratch)**

   * Accuracy
   * Precision
   * Recall
   * F1-score

---

## 🧠 Models Implemented from Scratch

The following algorithms were implemented **without using scikit-learn**:

* **Logistic Regression** (Gradient Descent)
* **K-Nearest Neighbors (KNN)**
* **Naive Bayes (Gaussian)**

Each model was evaluated using:

* Confusion Matrix
* Accuracy, Precision, Recall, F1-score

---

## 📉 Dimensionality Reduction

* **Principal Component Analysis (PCA)**
* Reduced features from **30 → 15**
* Applied PCA + Logistic Regression

```text
PCA reduced features from 30 → 15
```

---

## 🚀 Advanced Machine Learning Models

Implemented using **scikit-learn**:

* **Support Vector Machine (RBF Kernel)**
* **Random Forest Classifier**
* **Gradient Boosting Classifier**

### Pipeline Used

* Feature Scaling with `StandardScaler`
* Probability estimation enabled for ROC-AUC

---

## 📊 Model Performance Summary

### 🔹 ROC–AUC Scores

| Model             | ROC-AUC    |
| ----------------- | ---------- |
| SVM (RBF)         | **0.9962** |
| Random Forest     | 0.9894     |
| Gradient Boosting | 0.9896     |

### 🔹 Detailed Metrics

#### ✅ Support Vector Machine (Best Model)

* **Accuracy**: 0.9735
* **Precision**: 0.9737
* **Recall**: 0.9487
* **F1-score**: 0.9610

#### Random Forest

* Accuracy: 0.9646
* Precision: 0.9487
* Recall: 0.9487
* F1-score: 0.9487

#### Gradient Boosting

* Accuracy: 0.9469
* Precision: 0.9231
* Recall: 0.9231
* F1-score: 0.9231

📌 **Best Model Selected: SVM (RBF Kernel)**: SVM (RBF Kernel)**

---

## 🖼️ Results Snapshot

Below is a snapshot of the final evaluation results:

```
BEST MODEL: SVM
```

(Refer to the image in the repository for exact metric outputs and confusion matrices.)

---

## 🛠️ Technologies Used

* **Python**
* **NumPy, Pandas**
* **Matplotlib, Seaborn**
* **scikit-learn**
* **KaggleHub**

---

## 📁 Project Structure

```text
├── BreastCancer.ipynb
├── README.md
```

---

## ▶️ Necessary Modules to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap kagglehub
```

---

## 📌 Key Learnings

* Implemented ML algorithms **from scratch**
* Understood trade-offs between classical ML models
* Used **ROC-AUC** for reliable model comparison
* Demonstrated importance of **feature scaling for SVM**

---

## 🙌 Author

**Naman Jain**
AI Engineering | Machine Learning | Data Science

---

⭐ If you find this project useful, consider starring the repository!

