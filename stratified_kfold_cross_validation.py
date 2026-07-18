from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

X_imb, y_imb = make_classification(
    n_samples=1000, n_features=10, weights=[0.95, 0.05], random_state=42
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = LogisticRegression()
scores = cross_val_score(model, X_imb, y_imb, cv=skf, scoring='f1')

print("F1 Scores per fold:", scores)
print(f"Average F1: {scores.mean():.3f}")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_imb, y_imb)):
    test_ratio = np.bincount(y_imb[test_idx]) / len(test_idx)
    print(f"Fold {fold+1} class distribution in test set: {test_ratio}")