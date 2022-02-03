import numpy as np


# A self inflicting journey of me continiously forgetting again and again countless times to account for the biases of this network.
# I hope this will be of a great educational and comedic value to you as to why you should never code without writing down everything first


training_data = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]])
outputs = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]])

a_len = len(training_data[0])
z_len = 2
A_len = 2 
n = len(training_data)
o_len = len(outputs[0]) 


def init_weights_and_biases():
    weights1 = np.random.rand(z_len, a_len)
    weights2 = np.random.rand(A_len, z_len)
    bias1 = np.random.rand(z_len, 1)
    bias2 = np.random.rand(A_len, 1) 
    
    return weights1, weights2, bias1, bias2

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))


def softmax(x):
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


def delistify(a):
    if len(a) == 1:
        return a[0]
    else:
        return "it's not a single array list ;-;"

w1, w2, b1, b2 = init_weights_and_biases()


def front_prop():
    a = training_data
    z = np.array([[sigmoid(i[j] + b1[j][0]) for j in range(z_len)] for i in a.dot(w1.T)])
    # first compute A without putting it through the softmax function
    # then put A through the softmax function to get the correct A matrix
    A = np.array([softmax(i +  delistify(b2.T)) for i in z.dot(w2.T)])
    return z, A

z, A = front_prop() 


def info():
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

def back_prop():
    O = outputs
    a = training_data
    cost = np.average([sum(i) for i in np.square(np.subtract(O, A))])
    def z_gradient(k):
        return np.array([[sigmoid_derivative(a[i].dot(w1[j].T) + float(b1[j]))*a[i][k] for j in range(z_len)] for i in range(n)])

    
    dCdw = (-2/n)*sum([sum([(O[i][j] - A[i][j])*sigmoid_derivative(z[i])[j]*(z_gradient(0)[i].dot(w2[j].T)) for j in range(o_len)]) for i in range(n)])


        
    return cost, dCdw

cost, dCdw = back_prop()





 