
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

def build_ensemble_models():
    """Build and return classical Machine Learning classifiers."""
    models = {
        'SVM': SVC(kernel='rbf', probability=True, C=1.0),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42)
    }
    
    # Soft Voting Ensemble
    voting_clf = VotingClassifier(
        estimators=[('svm', models['SVM']), ('rf', models['RandomForest']), ('knn', models['KNN'])],
        voting='soft'
    )
    models['Ensemble_Voting'] = voting_clf
    return models

def evaluate_models_cv(X, y, n_splits=5):
    """Evaluate all benchmark classifiers using Stratified K-Fold Cross Validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = build_ensemble_models()
    benchmark_results = {}

    for name, model in models.items():
        accuracies = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            accuracies.append(accuracy_score(y_val, preds))
            
        benchmark_results[name] = np.mean(accuracies)
        print(f"Model: {name} | Mean CV Accuracy: {benchmark_results[name]*100:.2f}%")
        
    return benchmark_results
