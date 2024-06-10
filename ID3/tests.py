from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
from solver import ID3DecisionTree
import matplotlib.pyplot as plt

def find_best_depth():
    train_data = pd.read_csv('train_data.csv', sep=';')
    val_data = pd.read_csv('val_data.csv', sep=';')

    X_train = train_data[['age', 'weight', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc','smoke','alco','active']].values
    y_train = train_data['cardio'].values
    X_test = val_data[['age', 'weight', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc','smoke','alco','active']].values
    y_test = val_data['cardio'].values
    depths = {}
    for depth in range(1, 11):
        tree = ID3DecisionTree(X_train, y_train, max_depth=depth)
        predictions = [tree.predict(x) for x in X_test]
        score = 0
        for i in range(len(predictions)-1):
            if predictions[i] == y_test[i]:
                score += 1
        depths[depth] = score
    plt.bar(list(depths.keys()), depths.values(), color='g')
    plt.show()
    return depths 


def reports():
    train_data = pd.read_csv('train_data.csv', sep=';')
    test_data = pd.read_csv('test_data.csv', sep=';')

    X_train = train_data[['age', 'weight', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc','smoke','alco','active']].values
    y_train = train_data['cardio'].values
    X_test = test_data[['age', 'weight', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc','smoke','alco','active']].values
    y_test = test_data['cardio'].values

    tree = ID3DecisionTree(X_train, y_train, max_depth=3)

    predictions = [tree.predict(x) for x in X_test]
    print("ID3 Decision Tree (My Implementation) Report:")
    report = classification_report(y_test, predictions)
    print(report)

    sklearn_tree = DecisionTreeClassifier(max_depth=3)
    sklearn_tree.fit(X_train, y_train)

    sklearn_predictions = sklearn_tree.predict(X_test)

    print("\nSklearn Decision Tree Report:")
    sklearn_report = classification_report(y_test, sklearn_predictions)
    print(sklearn_report)


reports()
print(find_best_depth())