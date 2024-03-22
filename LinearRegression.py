import numpy as np
import tqdm
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in tqdm.tqdm(range(self.n_iterations)):  
            y_pred = np.dot(X, self.weights) + self.bias 
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def plot(self, X, y):
        y_pred = self.predict(X)
        plt.scatter(X, y)
        plt.plot(X, y_pred, color='red')
        plt.show()
        plt.grid(True)
 
if __name__ == "__main__": 
    X = np.array([[1, 2, 3, 4, 5]]).T   
    y = np.array([2, 3, 4, 5, 6])     
     
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    model.plot(X, y)
     
    print("Coefficients:", model.weights)
    print("Intercept:", model.bias)
    print("Predictions:", y_pred)
