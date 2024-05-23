import numpy as np
import random

# load dataset

class quadratic_cost(object):
    @staticmethod
    def fn(a, y):
        return np.linalg.norm(a,y)^2
    
    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)

class cross_entropy_cost(object):
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class Network(object):
    def __init__(self, sizes, cost=cross_entropy_cost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes [1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes [1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes [1:])]

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def feedforward(self, a):
    # returns output if a is input
    for w, b in zip(self.weights, self.biases):
        a = sigmoid(np.dot(w, a) + b) # a' = σ(wa + b)
    return a

def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None, lmbda=0.0, see_cost=False):
    # learning algorithm
    n = len(training_data)
    evaluation_cost = []
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            update_mini_batch(self, mini_batch, eta, lmbda, n)
        if see_cost:
            cost = total_cost(self, training_data, lmbda)
            evaluation_cost.append(cost)
            print(f"Cost of E.D: {cost}")
        if test_data:
            print(f"Epoch {j}: {evaluate(self, test_data)/100}%")
        else:
            print(f"Epoch {j} complete")

def update_mini_batch(self, mini_batch, eta, lmbda, n):
    # prep for bpg
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # compare
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backpropegate(self, x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

def backpropegate(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x # in layers
    activations = [x]
    zs = [] # list of all z vectors, layer by layer
    for w, b in zip(self.weights, self.biases):
        # calculate output based on previous layer
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # delta = cost_derivative(self, activations[-1], y) * sigmoid_prime(zs[-1])
    delta = self.cost.delta(zs[-1], activations[-1], y) 
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # iterate backwards starting from last hidden layer
    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)

def total_cost(self, data, lmbda):
    cost = 0.0
    for x, y in data:
        a = feedforward(self, x)
        cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
    return round(cost, 3)   

def evaluate(self, test_data):
    test_results = [(np.argmax(feedforward(self, x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)