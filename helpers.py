import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


binary_class_label = "home_team_win"
binary_class_label_complement = "away_team_win"

def train_test(table, test_size = 0.1):
    """
    Returns (train, test) of matches with a default 10% test split
    """
    test = table.sample(frac = test_size, random_state = 200)
    train = table.drop(test.index)
    return train, test

def evaluate(y_true, y_pred, suppress_print = False):
    """
    Helper to run different prediction evaluation functions
    """
    evaluators = {
        "rmse" : mean_squared_error(y_true, y_pred),
        "accuracy_score" : accuracy_score(y_true, y_pred) # Jaccard Index for binary classification
    }
    if not suppress_print:
        for name, evaluator in evaluators.items():
            print(f"{name} = {evaluator}")
    return evaluators

def logistic_regression(data, features, label = binary_class_label, test_size = 0.1, solver="newton-cholesky", suppress_print = False):
    """
    Train logistic regression on dataset.
    Performs a test_size % split of train/test data
    Returns the model, training data, testing data, and the evaluators
    """
    train, test = train_test(data, test_size)
    lr = LogisticRegression(solver = solver)
    lr.fit(train[features], train[label])
    pred = lr.predict(test[features])
    evaluators = evaluate(test[label], pred, suppress_print)
    return {"model": lr, "train_data": train, "test_data": test, "evaluators": evaluators}

def polynomial_features(X, degree = 2):
    """
    Given X is NxD
    Return N x (degree+1) x D array
    """
    N, D = X.shape
    powers = np.arange(degree + 1)[np.newaxis, :].T
    powers = np.repeat(powers, D, axis = 1)[np.newaxis, :]
    powers = np.repeat(powers, N, axis = 0)
    repeat = np.repeat(X[:, np.newaxis, :], degree + 1, axis = 1)
    return np.power(repeat, powers)

def poly_enhance_data(data, features, degree = 2, suppress_print = False):
    X = data[features].to_numpy()
    y = data[binary_class_label].to_numpy()
    N, D = X.shape
    X_poly = polynomial_features(X, degree)
    X_poly_2d = X_poly.reshape((N, D * (degree + 1)))
    poly_renamed_features = []
    for d in range(degree + 1):
        for i in range(X.shape[1]):
            poly_renamed_features.append(f"x_({i})^({d})")
    if not suppress_print:
        print(f"Input data shape = {X.shape}")
        print(f"Output data shape = {X_poly_2d.shape}")
        print(f"{X_poly_2d.shape[1] - D} new features generated of degree up to {degree}")
    poly_data = pd.DataFrame(data = X_poly_2d, columns = poly_renamed_features)
    poly_data[binary_class_label] = y
    return poly_data, poly_renamed_features

def train_test_error(lr, features, suppress_print = False):
    x_train = lr["train_data"][features]
    y_train = lr["train_data"][binary_class_label]

    x_test = lr["test_data"][features]
    y_test = lr["test_data"][binary_class_label]

    train_error = 1 - lr["model"].score(x_train, y_train)
    test_error = 1 - lr["model"].score(x_test, y_test)
    if not suppress_print:
        print(f"Train Error = {train_error}\nTest Error = {test_error}")
    return train_error, test_error

def plot_errors(train_error, test_error, parameters):
    """
    Plot change in train and test error for each parameter
    """
    plt.plot(parameters, train_error, label = "train_error")
    plt.plot(parameters, test_error, label = "test_error")
    plt.legend()
    plt.title("Train vs Test Error w.r.t parameter")
    plt.xlabel("Parameter")
    plt.ylabel("Error")
    plt.show()

def get_sorted_importances(lr, xtrain, ytrain, feature_names):
    importances_mean = permutation_importance(lr["model"], xtrain, ytrain)["importances_mean"]
    importances = pd.DataFrame(data = importances_mean, columns = ["mean_importance"])
    importances["feature_name"] = feature_names
    importances = importances.sort_values(by = "mean_importance", ascending=False)
    return importances

def plot_sorted_importances(sorted_importances, top_k = 10):
    data = sorted_importances.head(top_k)
    plt.bar(x = data["feature_name"], height = data["mean_importance"])
    plt.title("Mean Permutation Feature Importances")
    plt.xlabel("Feature Name")
    plt.ylabel("Mean Permutation Importance")
    plt.show()