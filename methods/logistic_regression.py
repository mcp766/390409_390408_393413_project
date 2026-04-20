import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier. 
    """
    

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        N, D = training_data.shape
        C = get_n_classes(training_labels)
        Y = label_to_onehot(training_labels)

        self.weights = np.zeros((D, C))

        for _ in range(self.max_iters):
            # Softmax
            logits = training_data @ self.weights
            logits -= logits.max(axis=1, keepdims=True)  # stabilité numérique
            exp = np.exp(logits)
            P = exp / exp.sum(axis=1, keepdims=True)

            # Gradient de la cross-entropy
            grad = training_data.T @ (P - Y) / N
            self.weights -= self.lr * grad

        pred_labels = onehot_to_label(P)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        logits = test_data @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        P = exp / exp.sum(axis=1, keepdims=True)
        pred_labels = onehot_to_label(P)
        return pred_labels
