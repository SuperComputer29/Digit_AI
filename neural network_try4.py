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
    exp_vector = np.exp(x)
    sm_vector = []

    for i in range(len(exp_vector)):
        sm_vector.append(exp_vector[i]/np.sum(exp_vector))

    return np.array(sm_vector)


def softmax_derivative(x):
   sm_vector = softmax(x)
   sm_vector_derivative = []

   for i in range(len(sm_vector)):
       sm_vector_derivative.append(sm_vector[i]*(1 - sm_vector[i]))

   return np.array(sm_vector_derivative)


def delistify(a): # Converts a single element list into a string containing that element
    if len(a) == 1:
        return np.array(a[0])
    else:
        return "it's not a single element list ;-;"

def listify(a): # Converts an element into an array containing that element
    return np.array([a])

w1, w2, b1, b2 = init_weights_and_biases()


def front_prop(w_1, w_2, b_1, b_2): # Front propogation
    a = training_data
    z = np.array([[sigmoid(i[j] + b_1[j][0]) for j in range(z_len)] for i in a.dot(w_1.T)])
    # first compute A without putting it through the softmax function
    # then put A through the softmax function to get the correct A matrix
    A = np.array([softmax(i +  delistify(b_2.T)) for i in z.dot(w_2.T)])
    return z, A
z, A = front_prop(w1, w2, b1, b2) 


def info(): # spits out information regarding the NN and also help you maintain your sanity
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
    epochs = 1000
    learning_rate = 0.1
    
    def cost_constant(e):
        return -2*sum(O[e - 1])
    
    A_inner_dx = np.array([w2[i][1]*sigmoid_derivative(delistify(a[1].dot(w1[1]) + b1[1]))*a[1][1] for i in range(A_len)])
    dAdw1 = softmax_derivative(z[1].dot(w2) + b2.T)
    
    
    def dCdw1(b, c, e): # e stands for example!!!
        A_inner_dx = np.array([w2[i][b - 1]*sigmoid_derivative(delistify(a[e - 1].dot(w1[b - 1]) + b1[b - 1]))*a[e - 1][c - 1] for i in range(A_len)])
        dAdw1 = delistify(softmax_derivative(z[e - 1].dot(w2) + b2.T))
        derivative = dAdw1.dot(A_inner_dx)

        return 2*derivative - 2*cost_constant(e)

    
    def dCdw2(b, c, e):
        if b == 1:
            return 2*delistify(softmax_derivative(z[e - 1].dot(w2) + b2.T))[0]*z[e - 1][c - 1] - 2*cost_constant(e)
        elif b == 2:
            return 2*delistify(softmax_derivative(z[e - 1].dot(w2) + b2.T))[1]*z[e - 1][c - 1] - 2*cost_constant(e) 


    def dCdb1(b, e):
        if b == 1:
            return 2*delistify(softmax_derivative(z[e - 1].dot(w2) + b2.T))[0] - 2*cost_constant(e)
        elif b == 2:
            return 2*delistify(softmax_derivative(z[e - 1].dot(w2) + b2.T))[1] - 2*cost_constant(e)               
    
    
    def dCdb2(b, e):
        if b == 1:
            return 2*delistify(softmax_derivative(z[e - 1].dot(w2) + b2.T))[0] - 2*cost_constant(e)
        else:
            return 2*delistify(softmax_derivative(z[e - 1].dot(w2) + b2.T))[1] - 2*cost_constant(e)


    for e in range(n):
        for i in range(epochs):
            front_prop(w1, w2, b1, b2)
            W1 = w1
            w1_adjustments = np.array([[learning_rate*dCdw1(b + 1, c + 1, e + 1) for c in range(a_len)] for b in range(z_len)])
            W1 = np.subtract(W1, w1_adjustments)
            
        for i in range(epochs):
            front_prop(w1, w2, b1, b2)
            W2 = w2
            w2_adjustments = np.array([[learning_rate*dCdw2(b + 1, c + 1, e + 1) for c in range(z_len)] for b in range(A_len)])
            W2 = np.subtract(W2, w2_adjustments)
            
        for i in range(epochs):
            front_prop(w1, w2, b1, b2)
            B1 = b1
            b1_adjustments = np.array([learning_rate*dCdb1(b + 1, e + 1) for b in range(z_len)])

            B1 = np.subtract(delistify(B1.T), b1_adjustments)
        
        for i in range(epochs):
            front_prop(w1, w2, b1, b2)
            B2 = b2
            b2_adjustments = np.array([learning_rate*dCdb2(b + 1, e + 1) for b in range(A_len)])
            B2 = np.subtract(delistify(B2.T), b2_adjustments)    

    
    Z, A_new = front_prop(W1, W2, listify(B1).T, listify(B2).T)
    
    return W1, W2, B1, B2, A_new

W1, W2, B1, B2, A_new = back_prop()

print(W1)
print(W2)
print(B1)
print(B2)
