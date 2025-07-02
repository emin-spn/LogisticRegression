from typing import List
import random
import math
class LogisticRegression:

    def __init__(self, learning_rate: float, epoch: int, batch_size: int):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.weights = []
        self.b = 0.0
        
    
    def fit(self, X: List[List[int]], y: List[List[int]]):
        for _ in range(len(X[0])):
            self.weights.append(random.gauss(0, math.sqrt(2 / len(X[0]))))
        
        logits = []
        for sample in X:
            logits.append(sum(self.weights * sample) + self.b) 
        
        numerators = []
        for i in logits:
            numerators.append(math.exp(i))
        
        denom = sum(numerators)

        softmax = []
        for n in numerators:
            softmax.append(n / denom)
    
        loss = 0
        for i in range(len(y)):
            loss -= y[i] * math.log(softmax[i])

        gradient = []

        #TODO find gradients, iteratively recalculate new weights, implement epoch and mini batch

        

    def predict(self, X: List[List[int]]):
        y_values = []
        for i in range(len(X)):
            y_values.append(W)