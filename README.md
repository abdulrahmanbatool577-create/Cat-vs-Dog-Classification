
#🐱🐶 Advanced Cat vs. Dog Image Classification & Multi-Model Benchmarking Framework
Computer Vision & Machine Learning Pipeline
🌟 Project Overview
An end-to-end Machine Learning pipeline designed for Cat vs. Dog image classification leveraging hybrid feature extraction (Color Statistics + HOG Descriptors), dimensionality reduction (PCA), and comparative evaluation across multiple classical supervised algorithms alongside soft-voting ensemble classifiers.

🚀 Key Technical Features
Hybrid Feature Extraction:
Color Features: Extracting Mean & Standard Deviation across BGR channels and 3D HSV Color Histograms.
Structural Features: Histogram of Oriented Gradients (HOG) for contour and shape representation.
Preprocessing & Feature Scaling: Automated StandardScaler and Principal Component Analysis (PCA) for feature space optimization.
Multi-Model Benchmarking: Comparative training & evaluation for:
Support Vector Machine (SVM)
Random Forest Classifier (RF)
K-Nearest Neighbors (KNN)
Decision Tree Classifier
Soft-Voting Ensemble Classifier (combining top-performing estimators).
Robust Evaluation: Evaluated via Stratified K-Fold Cross-Validation (5-Fold) for statistical reliability and precision.
⚙️ Tech Stack & Dependencies
Core Language: Python 3
Computer Vision: OpenCV (cv2), Scikit-Image (skimage)
Machine Learning & Analytics: Scikit-Learn, NumPy, Pandas
Visualization: Matplotlib / Seaborn
📂 Repository Structure
dataset_preprocessing.py: Preprocessing, HOG descriptors, and HSV color feature extractions.
model_training.py: Model architecture definition, Stratified K-Fold cross-validation, and soft-voting ensemble logic.
requirements.txt: Project dependencies and environment specs.
README.md: Technical documentation and architecture specification.
Model Benchmark Results & Accuracy
Model Architecture	Cross-Validation Accuracy	Status
Ensemble Soft Voting	87.60%	Best Model
Support Vector Machine (SVM)	84.50%	High Performance
Random Forest	82.10%	Stable
K-Nearest Neighbors (KNN)	78.30%	Baseline
Decision Tree	74.80%	Baseline
👩‍💻 Developer & Author
بتول جعوان | Batool Jaawan
Artificial Intelligence & Software Engineering Student
 
