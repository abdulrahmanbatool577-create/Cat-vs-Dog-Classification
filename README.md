
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    تدريب وتقييم النماذج المتعددة مع تحسين المعلمات (Hyperparameter Tuning)
    لرفع الدقة وأداء الـ Benchmarking وتجاوز نسبة 90%.
    """
    print("🚀 جاري بدء تدريب النماذج وتحسين الدقة...")

    # 1. إعداد النماذج بمعلمات محسّنة لمنع الـ Overfitting ورفع الكفاءة
    dt_model = DecisionTreeClassifier(max_depth=12, min_samples_split=5, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    svm_model = SVC(C=10, kernel='rbf', probability=True, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')

    # 2. نموذج التجميع المتقدم (Ensemble Soft Voting) مع تخصيص الأوزان للنماذج الأقوى
    ensemble_model = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('rf', rf_model),
            ('dt', dt_model)
        ],
        voting='soft',
        weights=[3, 2, 1]  # إعطاء أوزان أعلى لـ SVM و Random Forest لرفع النسبة النهائية فوق 90%
    )

    # قائمة النماذج لتقييمها
    models = {
        "Decision Tree (Tuned)": dt_model,
        "K-Nearest Neighbors (KNN)": knn_model,
        "Random Forest Classifier": rf_model,
        "Support Vector Machine (SVM)": svm_model,
        "Ensemble Soft Voting (Optimized)": ensemble_model
    }

    results = {}

    # التدريب والتقييم لكل نموذج
    for name, model in models.items():
        print(f"جاري تدريب النموذج: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        results[name] = round(acc, 2)
        print(f"✔ تم إنجاز {name} بدقة: {results[name]}%")

    print("\n========================================")
    print("📊 ملخص نتائج دقة النماذج (Model Benchmark):")
    for name, acc in results.items():
        print(f" - {name}: {acc}%")
    print("========================================")

    return results, ensemble_model

if __name__ == '__main__':
    print("✨ نظام التدريب وتحسين الدقة جاهز للتنفيذ.")
