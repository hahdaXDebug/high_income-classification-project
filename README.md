# An Implementation of Regularization and Classification in Classifying High Income Earners

## About
This project implements multiple machine learning methods in classifying and predicting high income earners using the Adult Census Income dataset of 1994. 

## Data Source
The dataset used is the Adult Census Income dataset of 1994, sourced from the UCI Machine Learning Repository

* **Observations:** 32,561
* **Target Variable:** `income` (<=50K or >50K)
* **Class Distribution:** Exploratory analysis reveals a significant baseline imbalance, with ~75.9% of observations falling into the lower-income bracket (<=50K) and ~24.1% in the higher-income bracket (>50K).

## Repository Contents

### 1. `High Income Classification.ipynb` (Exploratory & Modeling Notebook)
The Jupyter Notebook contains the comprehensive end-to-end analysis, including:
* **Exploratory Data Analysis (EDA):** Visualizing class distributions, feature correlations, and demographic proportions (e.g., age, education, and gender disparities).
* **Data Preprocessing:** Handling missing values (imputing modes for UCI's `?` placeholders), binning continuous variables like `age`, and simplifying categorical structures (e.g., grouping lower education levels).
* **Comprehensive Model Evaluation:** Training and tuning six distinct models:
  1. Logistic Regression (Baseline)
  2. Logistic Regression (L1/Lasso Penalty)
  3. Logistic Regression (L2/Ridge Penalty)
  4. Logistic Regression (ElasticNet)
  5. Random Forest Classifier
  6. Gradient Boosting Classifier

* **Feature Importance Analysis:** Extracting and comparing the top predictive features across all trained models to understand the primary indicators of income classification.

### 2. `high_income_gb.py` (Modular Execution Script)
A streamlined, production-ready Python script that isolates the optimal modeling strategy identified in the notebook. 
* Automatically fetches the dataset directly from the UCI repository.
* Replicates the exact preprocessing and feature engineering pipelines.
* Bypasses the lengthy `GridSearchCV` process by instantiating the recommended **Gradient Boosting Classifier** using the optimal tuned hyperparameters.
* Outputs a clean summary of performance metrics on the test set.

## Methodology & Model Selection

While all models were evaluated on standard metrics (Accuracy, Precision, Recall, F1-Score), special attention was given to the **Type 1 (False Positive)** and **Type 2 (False Negative)** error rates. 

In the context of household socio-economic classification, a Type 1 error (predicting a household is high-income when they are actually vulnerable or low-income) can lead to detrimental misclassifications—such as excluding an at-risk household from necessary policy support or subsidies. 

**Recommended Model:** `Gradient Boosting Classifier`
* **Optimal Hyperparameters:** `learning_rate=0.1`, `max_depth=5`, `n_estimators=200`
* **Rationale:** The tuned Gradient Boosting model achieved the most robust balance. It maintained a very low Type 1 error rate (~5.93%) while achieving the lowest Type 2 error rate (~32.59%) among all tested algorithms, making it highly reliable for practical, real-world deployment in policy analysis.


## Requirements to run
* **Python 3.13.3 or above:**

* **Libraries:**
  * NumPy
  * Pandas
  * Matplotlib
  * Seaborn
  * ScikitLearn
  * Warnings
 
To install the needed libraries for the script:
`pip install pandas numpy scikit-learn`
 

To run the script:
`python high_income_gb.py`

