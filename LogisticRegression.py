import numpy as np
import tqdm 

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None 
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in tqdm.tqdm(range(self.n_iterations)):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_cls
    
if __name__ == "__main__":
    X = np.array([[1, 2, 3, 4, 5]]).T
    y = np.array([0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Coefficients:", model.weights)
    print("Intercept:", model.bias)
    print("Predictions:", y_pred)



