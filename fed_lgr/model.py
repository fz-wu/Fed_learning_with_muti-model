import numpy as np
from scipy.special import expit
from scipy.special import softmax
class LogisticRegressionModel:
    """
    Implementation of Logistic Regression for Vertical Federated Learning.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, lr: float, label_num: int, seed: int = 12345):
        """
        Initialize the logistic regression model.

        Parameters:
        data (np.ndarray): The input data.
        lr (float): Learning rate for the model.
        seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.X = X
        self.Y = Y
        self.lr = lr
        self.label_num = label_num
        self.m, self.n = X.shape
        self.w = np.random.normal(loc=0.0, scale=1.0, size=(self.n, label_num))
        self.b = np.zeros((1, label_num))
        self.dw = None
        self.db = None
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - X.mean(axis=0)) / X.std(axis=0)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return expit(z)

    def softmax_fn(self, z):  # 多分类使用这个
        return softmax(z, axis=1)
    
    # def loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
    #     """
    #     Compute the logistic loss.
    #
    #     Parameters:
    #     y (np.ndarray): True labels.
    #     y_hat (np.ndarray): Predicted labels.
    #
    #     Returns:
    #     float: Computed loss.
    #     """
    #     epsilon = 1e-15
    #     loss = -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
    #     return loss
    def loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y * np.log(y_hat), axis=1))

    def gradients(self, X: np.ndarray, diff: np.ndarray) -> tuple:
        """Compute and return the gradients."""
        dw = (1 / self.m) * np.dot(X.T, diff)
        db = (1 / self.m) * np.sum(diff)
        return dw, db
    
    # def forward(self, x: np.ndarray) -> np.ndarray:
    #     """Forward pass."""
    #     self.x=x
    #     z = np.dot(self.x, self.w) + self.b
    #     return z
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        z = np.dot(self.x, self.w) + self.b
        return self.softmax_fn(z)
    
    # def compute_diff(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
    #     """Compute the difference between predicted and true labels."""
    #     y_hat = self.sigmoid(z)
    #     diff = y_hat - y
    #     return diff

    def compute_diff(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_hat = self.softmax_fn(z)
        return y_hat - y

    def logits(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.w) + self.b

    def compute_gradient(self,diff: np.ndarray):
  
        self.dw, self.db = self.gradients(self.x, diff)
        return self.dw,self.db
    
    def update_model_(self, y: np.ndarray) -> float:
        """Update the model parameters and return the loss."""
        self.w -= self.lr * self.dw
        self.b -= self.lr * self.db
        l = self.loss(y, self.sigmoid(np.dot(self.x, self.w) + self.b))
        return l
        
    def update_model(self):
        """Update the model parameters."""
        self.w -= self.lr * self.dw
        self.b -= self.lr * self.db
        
    def get_gradients(self) -> tuple:
        """Return the gradients if they exist."""
        if self.dw is not None and self.db is not None:
            return self.dw, self.db
        else:
            return None
    
    # def predict(self, X):
    #     if len(self.w.shape) == 1:
    #         self.w = self.w.reshape(-1, 1)
    #
    #
    #     preds = self.sigmoid(np.dot(X, self.w) + self.b)
    #     pred_class = [1 if i > 0.5 else 0 for i in preds]
    #     return np.array(pred_class).reshape(-1, 1)
    # def predict(self, X):
    #     z = np.dot(X, self.w) + self.b
    #     probs = self.sigmoid(z)
    #     return (probs >= 0.5).astype(int)
    def predict(self, X: np.ndarray):
        z = np.dot(X, self.w) + self.b
        probs = self.softmax_fn(z)
        return np.argmax(probs, axis=1).reshape(-1, 1)


        
   
