# Titanic Survival Prediction using Machine Learning

## Overview
This project applies Machine Learning techniques to predict the survival of Titanic passengers. It explores various data preprocessing, visualization, and model-building techniques to enhance predictive accuracy.

## Dataset
The dataset used in this project is the **Titanic dataset**, which contains information about passengers, including age, gender, class, and survival status.

## Features & Workflow
1. **Loading Dataset:** Read the Titanic dataset using pandas.
2. **Exploratory Data Analysis (EDA):**
   - Checking missing values, data types, and statistics.
   - Understanding feature distributions using histograms and boxplots.
3. **Data Visualization:**
   - Correlation heatmaps.
   - Survival rate analysis based on various features.
4. **Data Preprocessing:**
   - Handling missing values.
   - Encoding categorical variables.
   - Feature scaling using `StandardScaler`.
5. **Model Training:**
   - **Supervised Learning Models:**
   - Logistic Regression
   - Random Forest Classifier
   - K-Nearest Neighbors (KNN)
   - Support Vector Classifier (SVC)
   - Naive Bayes Classifier
   - **Unsupervised Learning Models:**
   - K-Means Clustering
   - DBSCAN Clustering
   - Hierachieral Clustering
7. **Model Evaluation:**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

## Libraries Used
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `seaborn` - Statistical graphics
- `scikit-learn` - Machine learning models and preprocessing

## Results
The models are evaluated based on accuracy, precision, recall, and F1-score. Logistic Regression, Random Forest, K-Nearest Neighbors, Support Vector Classifier, and Naive Bayes models were trained, and their performance was analyzed. Additionally, K-Means and DBSCAN clustering were used for unsupervised analysis.

## Installation & Usage
1. Clone the repository:
   
   git clone https://github.com/your-username/titanic-ml-prediction.git
   
   cd titanic-ml-prediction
   
2. Install dependencies:
   
   pip install -r requirements.txt
   
3. Run the Jupyter Notebook:
   
   jupyter notebook final_notebook.ipynb

## Future Improvements
Experimenting with other machine learning models.
Hyperparameter tuning for better accuracy.
Implementing deep learning models.

## Contributing
Feel free to fork this repository and make contributions.

## License
This project is open-source and available under the MIT License🚀.
