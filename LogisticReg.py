import numpy as np

class LogisticRegression:
    def __init__(self, learning=0.001, iterations = 1000):
        self.learning = learning
        self.iterations = iterations
        self.weight = None
        self.b = None

    def fit(self, X, y): 
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features) 
        self.b = 0

        for t in range(self.iterations):
            linear = np.dot(X, self.weight) + self.b
            y_pred = self.sigmoid(linear)


            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weight -= self.learning * dw
            self.b -= self.learning * db




    def predict(self, X):
        linear = np.dot(X, self.weight) + self.b
        y_pred = self.sigmoid(linear)
        y_pred_classes = [1 if i>0.5 else 0 for i in y_pred]
        return y_pred_classes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))