import numpy as np 


# A self inflicting journey of me continiously forgetting again and again countless times to account for the biases of this network.
# I hope this will be of a great educational and comedic value to you as to why you should never code without writing down everything first


training_data = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]])
outputs = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]])

a_len = len(training_data[0]) # Number of input neruons
z_len = 2 # Number ef hidden layer neurons
A_len = 2 # Number of output layer neurons
n = len(training_data) # Number of training examples
o_len = len(outputs[0]) # Number of expected outputs


def delistify(a): # Converts a single element list into a string containing that element
    if len(a) == 1:
        return np.array(a[0])
    else:
        return "it's not a single element list ;-;"

def listify(a): # Converts an element into an array containing that element
    return np.array([a])


def init_weights_and_biases():
    weights1 = np.random.rand(z_len, a_len).T
    weights2 = np.random.rand(A_len, z_len).T
    bias1 = delistify(np.random.rand(z_len, 1).T)
    bias2 = delistify(np.random.rand(A_len, 1).T)
    
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


w1, w2, b1, b2 = init_weights_and_biases()


def front_prop(w_1, w_2, b_1, b_2): # Front propogation
    a = training_data
    z = np.array([[sigmoid(i[j] + b_1[j]) for j in range(z_len)] for i in a.dot(w_1)])
    # first compute A without putting it through the softmax function
    # then put A through the softmax function to get the correct A matrix
    A = np.array([softmax(i +  b2) for i in z.dot(w_2)])
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
    epochs = 5000
    learning_rate = 0.0001


    def cost_constant(e):
        return -2*sum(O[e - 1])
    

    def dCdw1(b, c, e, w_1, w_2, z_1, b_1, b_2): # e stands for example!!!
        A_inner_dx = np.array([w_2[i][b - 1]*sigmoid_derivative(a[e - 1].dot(w_1.T[b - 1]) + b_1[b - 1])*a[e - 1][c - 1] for i in range(A_len)])
        dAdw1 = softmax_derivative(z_1[e - 1].dot(w_2) + b_2)
        derivative = dAdw1.dot(A_inner_dx)

        return 2*derivative - 2*cost_constant(e)

    
    def dCdw2(b, c, e, w_2, z_1, b_2):
        return 2*softmax_derivative(z_1[e - 1].dot(w_2) + b_2)[b - 1]*z_1[e - 1][c - 1] - 2*cost_constant(e) 


    def dCdb1(b, e, w_2, z_1, b_2):
        return 2*softmax_derivative(z_1[e - 1].dot(w_2) + b_2)[b - 1] - 2*cost_constant(e)               
    
    
    def dCdb2(b, e, w_2, z_1, b_2):
        return 2*softmax_derivative(z_1[e - 1].dot(w_2) + b_2)[b - 1] - 2*cost_constant(e)

    W1 = w1
    W2 = w2
    B1 = b1
    B2 = b2
    print(W1)
    for e in range(n):
        for i in range(epochs):
            Z_1, A_1 = front_prop(W1, W2, B1, B2)
            
            
            w1_adjustments = np.array([[learning_rate*dCdw1(b + 1, c + 1, e + 1, W1, W2, Z_1, B1, B2) for c in range(a_len)] for b in range(z_len)])
            W1 = np.subtract(W1, w1_adjustments.T)

            
            w2_adjustments = np.array([[learning_rate*dCdw2(b + 1, c + 1, e + 1, W2, Z_1, B2) for c in range(z_len)] for b in range(A_len)])
            W2 = np.subtract(W2, w2_adjustments.T)

            
            b1_adjustments = np.array([learning_rate*dCdb1(b + 1, e + 1, W2, Z_1, B2) for b in range(z_len)])
            B1 = np.subtract(B1.T, b1_adjustments)

            
            b2_adjustments = np.array([learning_rate*dCdb2(b + 1, e + 1, W2, Z_1, B2) for b in range(A_len)])
            B2 = np.subtract(B2.T, b2_adjustments)
            
            
    print(W1)
    Z, A_new = front_prop(W1, W2, B1, B2)
    
    return W1, W2, B1, B2, A_new

W1, W2, B1, B2, A_new = back_prop()



