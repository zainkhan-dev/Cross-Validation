# 📊 Cross Validation in Scikit-learn

A beginner-friendly machine learning project demonstrating two of the most commonly used cross-validation techniques in **Scikit-learn**:

* **K-Fold Cross Validation**
* **Stratified K-Fold Cross Validation**

The project uses **Logistic Regression** to train and evaluate models while showing how different validation strategies affect performance.

---

## 🚀 Project Overview

Cross-validation is an essential technique in machine learning for estimating how well a model will perform on unseen data.

This project compares:

* **K-Fold Cross Validation** on a balanced dataset.
* **Stratified K-Fold Cross Validation** on an imbalanced dataset.

It also demonstrates how stratification preserves class proportions in every fold, making evaluation more reliable for classification problems.

---

## ✨ Features

* Generate synthetic datasets using `make_classification()`
* Train a **Logistic Regression** classifier
* Perform **5-Fold Cross Validation**
* Perform **5-Fold Stratified Cross Validation**
* Calculate evaluation metrics
* Display accuracy for every fold
* Display F1-score for imbalanced classification
* Verify class distribution in each stratified fold

---

## 🛠️ Technologies Used

* Python 3
* NumPy
* Scikit-learn

---

## 📂 Project Structure

```
.
├── kfold_cross_validation.py
├── stratified_kfold_cross_validation.py
└── README.md
```

---

## 📖 What Each Script Does

### 1️⃣ K-Fold Cross Validation

This script:

* Creates a synthetic classification dataset.
* Splits the data into **5 folds**.
* Trains a Logistic Regression model on each fold.
* Evaluates the model using **Accuracy**.
* Prints:

  * Accuracy for each fold
  * Average Accuracy
  * Standard Deviation of Accuracy

### Example Output

```
Fold 1: Accuracy = 0.750
Fold 2: Accuracy = 0.775
Fold 3: Accuracy = 0.875
Fold 4: Accuracy = 0.775
Fold 5: Accuracy = 0.800

Average Accuracy: 0.795
Standard Deviation: 0.043
```

---

### 2️⃣ Stratified K-Fold Cross Validation

This script:

* Creates an **imbalanced dataset** (95% vs 5% classes).
* Uses **StratifiedKFold** to preserve class distribution.
* Evaluates the model using the **F1-score**, which is more suitable for imbalanced datasets.
* Displays:

  * F1-score for each fold
  * Average F1-score
  * Class distribution in every test fold

### Example Output

```
F1 Scores per fold: [0.4        0.42105263 0.4        0.30769231 0.47058824]
Average F1: 0.400
Fold 1 class distribution in test set: [0.95 0.05]
Fold 2 class distribution in test set: [0.95 0.05]
Fold 3 class distribution in test set: [0.945 0.055]
Fold 4 class distribution in test set: [0.945 0.055]
Fold 5 class distribution in test set: [0.945 0.055]
```

---

## 📊 Why Use Stratified K-Fold?

Regular K-Fold may create folds with uneven class distributions, especially for imbalanced datasets.

Stratified K-Fold ensures every fold contains approximately the same percentage of each class as the original dataset, resulting in more stable and reliable model evaluation.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/zainkhan-dev/Cross-Validation.git
cd Cross-Validation
```

### 2. Install dependencies

```bash
pip install numpy scikit-learn
```

### 3. Run the scripts

For K-Fold:

```bash
python kfold_cross_validation.py
```

For Stratified K-Fold:

```bash
python stratified_kfold_cross_validation.py
```
---

## 📚 Concepts Covered

* Cross Validation
* K-Fold Cross Validation
* Stratified K-Fold
* Logistic Regression
* Accuracy Score
* F1 Score
* Imbalanced Datasets
* Model Evaluation
* Scikit-learn Workflows

---

## 🎯 Learning Outcomes

After completing this project, you will understand:

* Why cross-validation is important.
* The difference between K-Fold and Stratified K-Fold.
* When to use Accuracy vs F1-score.
* How to evaluate machine learning models more reliably.
* How Scikit-learn simplifies model validation.

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository and submit a pull request.

---

## 📄 License

This project is intended for educational purposes and is free to use and modify.

---

## 👨‍💻 Author

**Zain Yaser Khan**

Computer Science Student | Aspiring AI Engineer | Learning Machine Learning with Scikit-learn
