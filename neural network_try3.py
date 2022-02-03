import numpy as np
import math as m
# inputs is an m x n matrix consisting of m examples with each example having n entries
# profit, working hours, employee satisfaction ==> x ε {0, 1}
training_data = [[0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]]
outputs = [[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]]

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*sigmoid(1-x)


def softmax(x):
    exp_array = []
    string = str(type(x[0])).split()[1]
    if string == "'numpy.ndarray'>":
        new = []
        for i in x:
            l = softmax(i)
            exp_array.clear()
            new.append(l)
        return np.array(new)
        
    else:           
        for k in x:
            exp_array.append(np.exp(k))
        return np.array([np.exp(i)/np.sum(exp_array) for i in x])

def softmax_derivative(x):
    return np.subtract(softmax(x), np.square(x))


input_len = len(training_data[0])
n = len(training_data)
z_len = 2
o_len = 3
layers = 3
# forpropogation

weights1 = np.random.rand(z_len, input_len)
z = np.array([[sigmoid(np.dot(weights1[i], training_data[j])) for i in range(len(weights1))] for j in range(len(training_data))])
weights2 = np.random.rand(o_len, z_len).T
prediction = softmax(np.dot(z, weights2))


def status():
    print(f"the neuron structure is {input_len}, {z_len}, {o_len}")
    print("the weights are in indexed in such a way that there are as many rows as the number of examples and as many columns as the number of weights")
    doYou = input("do you want to see the weights?")
    if doYou == "yes":
        weight_query = int(input(f"Which layer's weights do you wanna see[1 ==> {layers}]"))
        if weight_query == 1:
            print(weights1)
        elif weight_query == 2:
            print(weights2)
        else:
            print("incorrect layer")
    else:
        print(prediction)

# i stands for the ith training example and 
def maybe():
    dCdw = (2/n)*(sum(outputs) - sum(prediction))*sum([softmax_derivative(z[i]*weights2)*sigmoid_derivative(training_data*weights1[0][0]*training_data[0][0]) for i in range(len(z))])

# backpropogation

cost = (1/len(outputs))*np.sum([np.square(np.subtract(outputs[i], prediction[i])) for i in range(len(outputs))])
dCdw =2*sum([sum(outputs[i]) - sum(prediction[i]) for i in range(len(outputs))])
print(z[0])
print("    ")
print(weights1)
print("    ")
print(weights2)


