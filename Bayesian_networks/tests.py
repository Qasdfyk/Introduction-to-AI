from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from solver import Solver
import numpy as np


data = load_breast_cancer()
X = data.data
y = data.target

def evaluate_model_with_splits(X, y, n_splits=5, test_size=0.3):
    split_accuracies = []
    for i in range(n_splits):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=i)
        nb = Solver()
        nb.fit(X_train, y_train)
        y_val_pred = nb.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        split_accuracies.append(accuracy)
    
    split_mean = np.mean(split_accuracies)
    split_std = np.std(split_accuracies)
    
    return split_mean, split_std

def evaluate_model_with_kfold(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    nb = Solver()
    accuracies = cross_val_score(nb, X, y, cv=kf, scoring='accuracy')
    
    kfold_mean = np.mean(accuracies)
    kfold_std = np.std(accuracies)
    
    return kfold_mean, kfold_std


# Testy z różnymi podziałami na zbiór treningowy i walidacyjny
split_stds = []
split_means = []
for i in range(1, 9):
    split_mean, split_std = evaluate_model_with_splits(X, y, n_splits=5, test_size=(i / 10))
    split_means.append(split_mean)
    split_stds.append(split_std)

k1_values = range(10, 90, 10)

plt.figure(figsize=(10, 5))

# Plotting the Different Splits means
plt.subplot(1, 2, 1)
plt.plot(k1_values, split_means, marker='o', linestyle='-', color='b', label='Mean')
plt.xlabel('Test size in %')
plt.ylabel('Mean Score')
plt.title('Different Splits Mean Scores')
plt.legend()
plt.grid(True)

# Plotting the Different Splits standard deviations
plt.subplot(1, 2, 2)
plt.plot(k1_values, split_stds, marker='o', linestyle='-', color='r', label='Standard Deviation')
plt.xlabel('Test size in %')
plt.ylabel('Standard Deviation')
plt.title('Different Splits Standard Deviations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Testy z k-krotną walidacją krzyżową
kfold_stds = []
kfold_means = []
for i in range(2, 8):
    kfold_mean, kfold_std = evaluate_model_with_kfold(X, y, k=i)
    kfold_means.append(kfold_mean)
    kfold_stds.append(kfold_std)

k_values = range(2, 8)

plt.figure(figsize=(10, 5))

# Plotting the K-Fold means
plt.subplot(1, 2, 1)
plt.plot(k_values, kfold_means, marker='o', linestyle='-', color='b', label='Mean')
plt.xlabel('Number of Folds (k)')
plt.ylabel('Mean Score')
plt.title('K-Fold Mean Scores')
plt.legend()
plt.grid(True)

# Plotting the K-Fold standard deviations
plt.subplot(1, 2, 2)
plt.plot(k_values, kfold_stds, marker='o', linestyle='-', color='r', label='Standard Deviation')
plt.xlabel('Number of Folds (k)')
plt.ylabel('Standard Deviation')
plt.title('K-Fold Standard Deviations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Ostateczna jakość na zbiorze testowym
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=3)
nb = Solver()
nb.fit(X_train, y_train)
y_test_pred = nb.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred, target_names=data.target_names)

print(test_report)