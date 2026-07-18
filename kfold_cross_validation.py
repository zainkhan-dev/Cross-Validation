import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=10, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
    
    print(f"Fold {fold+1}: Accuracy = {score:.3f}")

print(f"\nAverage Accuracy: {np.mean(scores):.3f}")
print(f"Standard Deviation: {np.std(scores):.3f}")