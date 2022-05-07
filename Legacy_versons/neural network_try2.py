import random as r
import numpy as np


# We'll have 4 layers; an input layer, 2 hidden layers and an output layer


inputs = [3,5]

a2 = []
a3 = []
a4 = []

neurons = [a2,a3,a4]

w1 = []
w2 = []
w3 = []

weights = [w1, w2, w3]

na = 16 #  the number of neurons in the 2nd layer
nb = 16 #  the number of neurons in the 3rd layer
nc = 10 #  the number of neurons in the output layer


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def dotProduct(A, B, n):
    if len(A) == len(B):
        if n - 1 == 0:
            return A[0]*B[0]
        else:
            return A[n - 1]*B[n - 1] + dotProduct(A, B, n - 1)
    else:
        print(f"Both the arrays {A} and {B} have different number of elements")


def initializingWeights():
    for i in range(na):
        a = []
        w1.append(a)
        for j in range(len(inputs)):
            b = r.randint(-10,10)
            w1[i].append(b)
    for i in range(nb):
        a = []
        w2.append(a)
        for j in range(na):
            b = r.randint(-10,10)
            w2[i].append(b)
    for i in range(nc):
        a = []
        w3.append(a)
        for j in range(nb):
            b = r.randint(-10,10)
            w3[i].append(b)
    
initializingWeights()

def initializingNeurons():
    for i in range(na):
        b = sigmoid(dotProduct(inputs, w1[i], len(inputs)))
        a2.append(b)
    for i in range(nb):
        b = sigmoid(dotProduct(a2, w2[i], len(a2)))
        a3.append(b)
    for i in range(nc):
        b = sigmoid(dotProduct(a3, w3[i], len(a3)))
        a4.append(b)

initializingNeurons()    


