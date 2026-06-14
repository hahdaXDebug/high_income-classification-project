import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Loads the Adult Census dataset directly from the UCI repository."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    # Strip whitespaces from strings while reading
    df = pd.read_csv(url, names=columns, skipinitialspace=True)
    return df

def preprocess_data(df):
    """Handles missing values, imputations, and groupings."""
    print("--- Starting Preprocessing ---")
    df_processed = df.copy()
    
    # Handle missing values: UCI Census uses '?' for unknowns
    df_processed.replace('?', np.nan, inplace=True)
    
    # Simple mode imputation
    df_processed['workclass'].fillna(df_processed['workclass'].mode()[0], inplace=True)
    df_processed['occupation'].fillna(df_processed['occupation'].mode()[0], inplace=True)
    df_processed['native-country'].fillna(df_processed['native-country'].mode()[0], inplace=True)
    
    # Simplify Education: Grouping lower education levels together
    edu_map = {
        'Preschool': 'Dropout', '1st-4th': 'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout',
        '9th': 'Dropout', '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout',
        'HS-grad': 'HS-grad', 'Some-college': 'College', 'Assoc-voc': 'College', 'Assoc-acdm': 'College',
        'Bachelors': 'Bachelors', 'Masters': 'Masters', 'Prof-school': 'Prof-school', 'Doctorate': 'Doctorate'
    }
    df_processed['education_simple'] = df_processed['education'].map(edu_map)
    
    # Simplify Age Groups 
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                     bins=[0, 25, 45, 65, 100], 
                                     labels=['Young', 'Middle-Aged', 'Senior', 'Old'])
    
    return df_processed

def prepare_features(df):
    """Encodes categorical variables and returns the Feature Matrix X and Target y."""
    print("--- Extracting and Encoding Features ---")
    df_encoded = df.copy()
    
    # Target encoding (>50K becomes 1, <=50K becomes 0)
    le = LabelEncoder()
    y = le.fit_transform(df_encoded['income']) 
    
    # Define categorical and numerical columns
    categorical_cols = ['workclass', 'education_simple', 'marital-status', 'occupation', 
                        'relationship', 'race', 'sex', 'native-country', 'age_group']
    numerical_cols = ['fnlwgt', 'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    # Label Encoding for categorical columns to make them usable by tree models
    for col in categorical_cols:
        df_encoded[col + '_enc'] = le.fit_transform(df_encoded[col].astype(str))
    
    # Select final feature list
    feature_names = numerical_cols + [col + '_enc' for col in categorical_cols]
    X = df_encoded[feature_names]
    
    return X, y, feature_names

def evaluate_model(y_true, y_pred, model_name, dataset_type):
    """Calculates and returns performance metrics including Type 1 & Type 2 errors."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    type1_error = fp / (tn + fp) if (tn + fp) > 0 else 0  # False Positive Rate
    type2_error = fn / (tp + fn) if (tp + fn) > 0 else 0  # False Negative Rate
    
    metrics = {
        "Model": model_name,
        "Dataset": dataset_type,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity (%)": specificity * 100,
        "Type 1 Error FPR (%)": type1_error * 100,
        "Type 2 Error FNR (%)": type2_error * 100
    }
    return metrics

def main():
    # 1. Load Data
    print("Loading data from UCI repository...")
    census_data = load_data()
    
    # 2. Preprocess
    processed_data = preprocess_data(census_data)
    
    # 3. Feature Engineering
    X, y, feature_names = prepare_features(processed_data)
    
    # 4. Train-Test Split & Scaling
    print("--- Splitting & Scaling Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Model Training (Using optimal GridSearch params from notebook)
    print("--- Training Recommended Gradient Boosting Model ---")
    print("Hyperparameters: learning_rate=0.1, max_depth=5, n_estimators=200")
    
    grad_boost = GradientBoostingClassifier(
        learning_rate=0.1, 
        max_depth=5, 
        n_estimators=200, 
        random_state=42
    )
    grad_boost.fit(X_train_scaled, y_train)
    
    # 6. Predictions & Testing
    print("--- Generating Predictions ---")
    y_test_pred_gb = grad_boost.predict(X_test_scaled)
    
    # 7. Results
    print("\n--- Model Test Evaluation ---")
    results = evaluate_model(y_test, y_test_pred_gb, "Gradient Boosting", "Test")
    
    # Display Results DataFrame
    metrics_df = pd.DataFrame([results])
    print("\nPerformance Metrics:")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()