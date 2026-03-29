import numpy as np

# --------------------------------- MODEL ---------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleModel:
    def __init__(self):
        self.W1 = np.array([[0.5, 0.2], [0.3, 0.8]]) 
        self.b1 = np.zeros((2,1)) 
        self.W2 = np.array([[0.6, 0.4]]) 
        self.b2 = np.zeros((1, 1)) 
        self.lr = 50
        self.y_hat = 0
        self.x = np.zeros((2,1))

    def forward(self, x):
        self.x = x # (2x1)
        self.z1 = self.W1 @ x + self.b1 
        self.h1 = sigmoid(self.z1)  
        self.z2 = self.W2 @ self.h1 + self.b2 
        self.y_hat = sigmoid(self.z2) 

        return self.y_hat
    
    def updateState(self, y):
        # delta = dL/dz2
        delta = 2 * (y - self.y_hat) * self.y_hat * (1 - self.y_hat)
        self.W1 = self.W1 + self.lr * delta * (self.W2.T * self.h1 * (1 - self.h1)) @ self.x.T
        self.b1 = self.b1 + self.lr * delta * self.W2.T * self.h1 * (1 - self.h1)
        self.W2 = self.W2 + self.lr * delta * self.h1.T
        self.b2 = self.b2 + self.lr * delta
    
    def changeLr(self, lr):
        """For the lr experiment"""
        self.lr = lr
