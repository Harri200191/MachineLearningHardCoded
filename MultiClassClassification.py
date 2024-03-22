import numpy as np
import tqdm 

class MultiClassClassification:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        y_encoded = np.eye(n_classes)[y]
        for _ in tqdm.tqdm(range(self.n_iterations)):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(linear_model)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y_encoded))
            db = (1/n_samples) * np.sum(y_pred - y_encoded)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(linear_model)
        y_pred_cls = np.argmax(y_pred, axis=1)
        return y_pred_cls
    
if __name__ == "__main__":
    X = np.array([[1, 2, 3, 4, 5]]).T
    y = np.array([2, 1, 2, 1, 0])
    model = MultiClassClassification(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Coefficients:", model.weights)
    print("Intercept:", model.bias)
    print("Predictions:", y_pred)