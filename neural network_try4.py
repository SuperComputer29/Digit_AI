import numpy as np


# A self inflicting journey of me continiously forgetting again and again countless times to account for the biases of this network.
# I hope this will be of a great educational and comedic value to you as to why you should never code without writing down everything first


training_data = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]])
outputs = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]])

a_len = len(training_data[0]) # Number of input neruons
z_len = 2 # Number of hidden layer neurons
A_len = 2 # Number of output layer neurons
n = len(training_data) # Number of training examples
o_len = len(outputs[0]) # Number of expected outputs


def init_weights_and_biases():
    weights1 = np.random.rand(z_len, a_len)
    weights2 = np.random.rand(A_len, z_len)
    bias1 = np.random.rand(z_len, 1)
    bias2 = np.random.rand(A_len, 1) 
    
    return weights1, weights2, bias1, bias2

def sigmoid(x): # The first activation function that transforms the dot product into the hidden layer
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x): # The derivative of the sigmoid function
    return sigmoid(x)*(1 - sigmoid(x))


def softmax(x): # The second activation function
    exp_x = []
    for i in range(len(x)):
        exp_x.append(np.exp(x[i]))

    return np.array([np.exp(x[i])/sum(exp_x) for i in range(len(x))])


def softmax_derivative(x):
    nx = len(x)
    one_vector = []
    for i in range(nx):
        one_vector.append(1)
    one_vector = np.array(one_vector)
    return softmax(x).dot(np.subtract(1, softmax(x)))


def delistify(a): # Converts a single element list into a string containing that element
    if len(a) == 1:
        return a[0]
    else:
        return "it's not a single element list ;-;"

w1, w2, b1, b2 = init_weights_and_biases()


def front_prop(): # Front propogation
    a = training_data
    z = np.array([[sigmoid(i[j] + b1[j][0]) for j in range(z_len)] for i in a.dot(w1.T)])
    # first compute A without putting it through the softmax function
    # then put A through the softmax function to get the correct A matrix
    A = np.array([softmax(i +  delistify(b2.T)) for i in z.dot(w2.T)])
    return z, A

z, A = front_prop() 


def info(): # I'll spit out information regarding the NN and also help you maintain your sanity
    print("    a      ")
    print("                  ")
    print(training_data)
    print("                  ")
    print("           w1      ")
    print("                  ")
    print(w1)
    print("                  ")
    print("           b1      ")
    print("                  ")
    print(b1)
    print("                  ")
    print("           z      ")
    print("                  ")
    print(z)
    print("                  ")
    print("           w2      ")
    print("                  ")
    print(w2)
    print("                  ")
    print("           b2      ")
    print("                  ")
    print(b2)
    print("                  ")
    print("           A      ")
    print("                  ")
    print(A) 

def back_prop(): # Back propogation
    O = outputs
    a = training_data
    cost2 = np.average([sum(i) for i in np.square(np.subtract(O, A))])
    epochs = 1
    learning_rate = 0.1

    # vectors

    # w1_v = w1.reshape([a_len*z_len, 1])
    # w2_v = w1.reshape([z_len*A_len, 1])
    # b1_v = b1
    # w1_v = b2

    def w1(b, c, example):
        return -2*(O-A)*softmax_derivative(z[example].dot(w2) + b2)*w1[b][c]*sigmoid_derivative(a[example].dot(w1) + b1)*w1[b][c]


    def w2(b, c, example):
        return -2*(O-A)*softmax_derivative(z[example].dot(w2) + b2)*w2[b][c]


    def b1(b, example):
        return -2*(O-A)*softmax_derivative(z[example].dot(w2) + b2)[b]*sigmoid_derivative(a[example].dot(w1[b]) + b1[b])


    def b2(b, example):
        return -2*(O-A)*softmax_derivative(z[example].dot(w2) + b2)[b]
        

    for k in range(epochs): # i is already taken :(
        for example in range(n):
            front_prop()
            w1_change = [[w1(i, j, example) for j in range(a_len)] for i in range(z_len)]
            w2_change = [[w2(i, j, example) for j in range(z_len)] for i in range(A_len)]
            b1_change = [b1[i, example] for i in range(z_len)]
            b2_change = [b2[i, example] for i in range(A_len)]

            w1 -= learning_rate*w1_change
            w2 -= learning_rate*w2_change
            b1 -= learning_rate*b1_change
            b2 -= learning_rate*b2_change

        
    return w1, w2, b1, b2

W1, W2, B1, B2 = back_prop()





 