import random as r


layers = []
weights = []
# activations/neutrons are indexed in the form a_mn where 
# m is the position of the neuron that is in it's layer and where n is the layer it's in
# weights are indexed in the form w_ab where a is the position of the neuron from where
# the weight stems out and b is the b is the bth weight from that aforementioned neuron
def createActivations():
    n = int(input("How many layers do you want?: "))
    for i in range(n):
        m = int(input(f"How many activations/neurons do you want in the {i+1}th layer?"))
        layers.append([])
        for j in range(m):
            a_n = f"a_{i+1}{j+1}"
            layers[i].append(a_n)
           

createActivations()

# Metadata about the neural network's activations
n = len(layers)
neuron_n = []
for i in range(n):
    neuron_n.append(len(layers[i]))

def activationValues(a):
    for i in n:
        for a in i:
            print("yea")
            print(i.index(a))


def intializingWeights():
    withNumbers = (input("Do you want to intitialize the weights with numbers?[Y/n]:")).lower()
    for i in range(n):
        if i == n - 1:
            return
        else:
            weights.append([])
            for a in range(neuron_n[i]):
                for b in range(neuron_n[i + 1]):
                    if withNumbers == "y":
                        w_ab = f"w_{a+1}{b+1}"
                        w_ab = r.uniform(0,1)
                        weights[i].append(w_ab)
                    elif withNumbers == "n":
                        w_ab = f"w_{a+1}{b+1}"                    
                        weights[i].append(w_ab)
                        
intializingWeights()
print(weights)

def initializingNeurons(I):
    layer_1 = layers[0]
    for i in range(len(layer_1)): # sets the first layer
        layer_1[i] = I[i]
    for i in range(1, len(layers)):
        for j in range(len(layers[i])):
            index =  int(list(layers[i][j])[3])
            for k in weights[i-1]:
                for l in k:
                    print(len(list(l)))
            


initializingNeurons([3,2,3,4])
print(layers)
            
#convert the layers and weights arrays into dictionaries


