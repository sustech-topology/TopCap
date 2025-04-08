import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def parallel_gaussian_svm_cv(csv_file, n_jobs=-1):
    """
    Accelerate Gaussian SVM 5-fold cross-validation using parallel computation.
    
    Parameters:
        csv_file: Path to the CSV file.
        n_jobs: Number of parallel tasks (-1 indicates using all CPU cores).
    
    Returns:
        The average cross-validation accuracy and the results for each fold.
    """
    try:
        # 1. Read data
        data = pd.read_csv(csv_file, header=None)
        X = data.iloc[:, [2, 3]].values  # Features are in the 3rd and 4th columns.
        y = data.iloc[:, 4].values       # Labels are in the 5th column.

        # 2. Check if data is binary classification
        if len(np.unique(y)) != 2:
            raise ValueError("Labels must be binary classification data!")

        # 3. Create pipeline (standardization + SVM)
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', gamma='scale', random_state=42)
        )

        # 4. Stratified 5-fold cross-validation (parallel computation)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X, y,
            cv=skf,
            scoring='accuracy',
            n_jobs=n_jobs  # Run cross-validation in parallel
        )

        # 5. Output results
        print(f"5-fold cross-validation accuracy: {cv_scores}")
        print(f"Mean accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        return {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'all_scores': cv_scores
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    csv_path = "../data/Sample_TIMIT_noise0_arr.csv"
    print(csv_path)
    results = parallel_gaussian_svm_cv(csv_path, n_jobs=6)
