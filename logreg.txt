from typing import List
import random
import math

class LogisticRegression:

    def __init__(self, learning_rate: float, epoch: int, batch_size: int):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.weights = []
        self.b = []

    def fit(self, X: List[List[float]], y: List[int]):
        num_features = len(X[0])
        num_classes = len(set(y))
        self.weights = [[random.gauss(0, math.sqrt(2 / num_features)) for _ in range(num_features)] for _ in range(num_classes)]
        self.b = [0.0] * num_classes

        for _ in range(self.epoch):
            batches = self._build_mini_batches(X, y, self.batch_size)
            for batch_X, batch_y in batches:
                logits = self._dot_product(batch_X, self.weights)
                logits = [[l + b_i for l, b_i in zip(logit, self.b)] for logit in logits]
                softmax = self._get_softmax(logits)
                loss = [[0.0] * num_classes for _ in range(len(batch_y))]
                for i, label in enumerate(batch_y):
                    loss[i][label] = 1 - softmax[i][label]
                grad_w, grad_b = self._compute_gradients(batch_X, loss)
                self.weights = [[w - self.learning_rate * gw for w, gw in zip(ws, grad_ws)] for ws, grad_ws in zip(self.weights, grad_w)]
                self.b = [b - self.learning_rate * gb for b, gb in zip(self.b, grad_b)]

    def predict(self, X: List[List[float]]):
        logits = self._dot_product(X, self.weights)
        logits = [[l + b_i for l, b_i in zip(logit, self.b)] for logit in logits]
        softmax = self._get_softmax(logits)
        return [self._argmax(prob) for prob in softmax]

    @staticmethod
    def _get_softmax(logits):
        exp_logits = []
        for logit in logits:
            max_logit = max(logit)
            exp_row = [math.exp(l - max_logit) for l in logit]
            exp_logits.append(exp_row)
        
        softmax = []
        for exp_row in exp_logits:
            total_exp = sum(exp_row)
            softmax_row = [exp / total_exp for exp in exp_row]
            softmax.append(softmax_row)
        
        return softmax


    @staticmethod
    def _dot_product(X, W):
        return [[sum(x * w for x, w in zip(row, col)) for col in zip(*W)] for row in X]

    @staticmethod
    def _build_mini_batches(X, y, batch_size):
        mini_batches = []
        combined = list(zip(X, y))
        random.shuffle(combined)
        for i in range(0, len(combined), batch_size):
            mini_batch = combined[i:i + batch_size]
            mini_batches.append(( [x for x, _ in mini_batch], [y for _, y in mini_batch] ))
        return mini_batches

    @staticmethod
    def _compute_gradients(self, X, loss):
        num_samples = len(X)
        num_features = len(X[0])
        num_classes = len(loss[0])

        grad_w = [[0.0] * num_features for _ in range(num_classes)]
        grad_b = [0.0] * num_classes

        for i in range(num_samples):
            for j in range(num_classes):
                for k in range(num_features):
                    grad_w[j][k] += X[i][k] * loss[i][j]
                grad_b[j] += loss[i][j]
        
        grad_w = [[gw / num_samples for gw in grad_ws] for grad_ws in grad_w]
        grad_b = [gb / num_samples for gb in grad_b]
        return grad_w, grad_b

    @staticmethod
    def _argmax(lst):
        return lst.index(max(lst))