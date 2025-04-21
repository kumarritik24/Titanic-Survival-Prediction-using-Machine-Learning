# 🚢 Titanic Survival Prediction using Machine Learning

🎯 Built an end-to-end machine learning pipeline to predict passenger survival in the Titanic dataset. Applied classification models (SVM, Random Forest, Logistic Regression) and achieved strong results using feature engineering, visualization, and evaluation metrics.

---

## 📁 Project Overview

This project applies classic ML techniques to predict the survival of Titanic passengers. It involves:

- Cleaning and preprocessing data
- Feature extraction and transformation
- Exploratory data analysis (EDA)
- Applying supervised & unsupervised models
- Model evaluation using classification metrics

---

## 📊 Dataset

The dataset used is the **Titanic dataset**, which includes information like:
- Passenger ID, Name
- Age, Gender, Class
- Fare, Embarked Port
- Survival status

---

## 🔍 Features & Workflow

<details>
  <summary>📦 Data Processing & Feature Engineering</summary>

- Handled missing values and outliers
- Extracted features like `Title`, `FamilySize`, `IsAlone`
- One-hot encoded categorical variables
- Correlation heatmap + feature importance

</details>

<details>
  <summary>📈 Exploratory Data Analysis (EDA)</summary>

- Distribution plots for Age, Fare, Class
- Survival rate by Gender, Class, Embarked
- Cross-tab visualizations
</details>

<details>
  <summary>🤖 Model Training & Evaluation</summary>

- **Supervised Models**:
  - Logistic Regression
  - Random Forest Classifier
  - SVM
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
- **Unsupervised Models**:
  - K-Means Clustering
  - DBSCAN Clustering
- **Model Metrics**:
  - Accuracy, F1-score, Recall
  - Confusion Matrix
  - ROC-AUC Score

</details>

---

## 🧰 Tools & Libraries

- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn` – Data visualization
- `scikit-learn` – Modeling & preprocessing
- `jupyter notebook` – Development interface

---

## 🧪 Results

The models were evaluated on:
- Accuracy
- Precision
- Recall
- F1-score

💡 Achieved high predictive performance using Random Forest and SVM models with properly tuned hyperparameters.

---

## ⚙️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/kumarritik24/Titanic-Survival-Prediction-using-Machine-Learning.git
cd Titanic-Survival-Prediction-using-Machine-Learning

# Install required packages
pip install -r requirements.txt

# Run the notebook
jupyter notebook titanic-ml.ipynb
