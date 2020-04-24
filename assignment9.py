import numpy as np
import pandas as pd

def load_data(filePath):
    data=pd.read_csv(filePath)

    res=pd.DataFrame()
    res['number']=data.iloc[:,-1]
    # print(y.head())

    data=data.drop(data.columns[-1],axis=1)

    x=[]
    y=[]
    for i in range(0,data.shape[0]):
        x.append(list(data.iloc[i,:]))
    for i in range(0,res.shape[1]):
        y.append(res.iloc[:,i])


def update_weights_perceptron(X, Y, weights, bias, lr):
    n=X.shape[0]
    m=X.shape[1]
    lt = Y.shape[1]
    for i in range(2000):
        z=np.matmul(weights.T,X)+bias
        A=np.exp(z-np.max(z))
        activation=A/A.sum()

        cost = -(1. / lt) * (np.sum(np.multiply(np.log(activation), Y)) + np.sum(np.multiply(np.log(1 - activation), (1 - Y))))

        dW = (1 / m) * np.matmul(X, (activation - Y).T)
        db = (1 / m) * np.sum(activation - Y, axis=1, keepdims=True)

        updated_weights = weights - lr * dW
        updated_bias = bias - lr * db

    return updated_weights, updated_bias

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t

def relu(x):
    if x>0:
        return x
    else:
        return 0

def update_weights_single_layer(X, Y, weights, bias, lr):
    n_x = X.shape[0]
    m = X.shape[1]
    lt=Y.shape[1]
    n_h = 10
    for i in range(2000):
        Z1 = np.matmul(weights[0], X) + bias[0]
        A1 = sigmoid(Z1)
        Z2 = np.matmul(weights[1], A1) + bias[1]
        A2 = sigmoid(Z2)


        cost = -(1. / lt) * (np.sum(np.multiply(np.log(A2), Y)) + np.sum(np.multiply(np.log(1 - A2), (1 - Y))))
        dZ2 = A2 - Y
        dW2 = (1. / m) * np.matmul(dZ2, A1.T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(weights[1].T, dZ2)
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        dW1 = (1. / m) * np.matmul(dZ1, X.T)
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

        weights[1] = weights[1] - lr* dW2
        bias[1] = bias[1] - lr * db2
        weights[0] = weights[0] - lr * dW1
        bias[0] = bias[0] - lr * db1
        updated_weights=[]
        updated_weights.append(weights[0])
        updated_weights.append(weights[1])
        updated_bias=[]
        updated_bias.append(bias[0])
        updated_bias.append(bias[1])

    return updated_weights, updated_bias



def update_weights_double_layer(X, Y, weights, bias, lr):
    n_x = X.shape[0]
    m = X.shape[1]
    lt=Y.shape[1]
    n_h = 10
    for i in range(2000):
        Z1 = np.matmul(weights[0], X) + bias[0]
        A1 = sigmoid(Z1)
        Z2 = np.matmul(weights[1], A1) + bias[1]
        A2 = sigmoid(Z2)
        Z3 = np.matmul(weights[2], A2) + bias[2]
        A3 = sigmoid(Z3)


        cost = -(1. / lt) * (np.sum(np.multiply(np.log(A3), Y)) + np.sum(np.multiply(np.log(1 - A3), (1 - Y))))
        dZ3 = A3 - Y
        dW3 = (1. / m) * np.matmul(dZ3, A2.T)
        db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.matmul(weights[2].T, dZ3)
        dZ2 = dA2 * sigmoid(Z2) * (1 - sigmoid(Z2))
        dW2 = (1. / m) * np.matmul(dZ2, A1.T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(weights[1].T, dZ2)
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        dW1 = (1. / m) * np.matmul(dZ1, X.T)
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

        weights[2] = weights[2] - lr* dW3
        bias[2] = bias[2] - lr * db3
        weights[1] = weights[1] - lr* dW2
        bias[1] = bias[1] - lr * db2
        weights[0] = weights[0] - lr * dW1
        bias[0] = bias[0] - lr * db1
        updated_weights=[]
        updated_weights.append(weights[0])
        updated_weights.append(weights[1])
        updated_weights.append(weights[2])
        updated_bias=[]
        updated_bias.append(bias[0])
        updated_bias.append(bias[1])
        updated_bias.append(bias[2])

    return updated_weights, updated_bias





def update_weights_double_layer_act(X, Y, weights, bias, lr, activation):
    n_x = X.shape[0]
    m = X.shape[1]
    lt = Y.shape[1]
    n_h = 10
    if activation == 'sigmoid':
        for i in range(2000):
            Z1 = np.matmul(weights[0], X) + bias[0]
            A1 = sigmoid(Z1)
            Z2 = np.matmul(weights[1], A1) + bias[1]
            A2 = sigmoid(Z2)
            Z3 = np.matmul(weights[2], A2) + bias[2]
            A3 = sigmoid(Z3)

            cost = -(1. / lt) * (np.sum(np.multiply(np.log(A3), Y)) + np.sum(np.multiply(np.log(1 - A3), (1 - Y))))
            dZ3 = A3 - Y
            dW3 = (1. / m) * np.matmul(dZ3, A2.T)
            db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

            dA2 = np.matmul(weights[2].T, dZ3)
            dZ2 = dA2 * sigmoid(Z2) * (1 - sigmoid(Z2))
            dW2 = (1. / m) * np.matmul(dZ2, A1.T)
            db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.matmul(weights[1].T, dZ2)
            dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
            dW1 = (1. / m) * np.matmul(dZ1, X.T)
            db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

            weights[2] = weights[2] - lr * dW3
            bias[2] = bias[2] - lr * db3
            weights[1] = weights[1] - lr * dW2
            bias[1] = bias[1] - lr * db2
            weights[0] = weights[0] - lr * dW1
            bias[0] = bias[0] - lr * db1
            updated_weights = []
            updated_weights.append(weights[0])
            updated_weights.append(weights[1])
            updated_weights.append(weights[2])
            updated_bias = []
            updated_bias.append(bias[0])
            updated_bias.append(bias[1])
            updated_bias.append(bias[2])

    if activation == 'tanh':
        for i in range(2000):
            Z1 = np.matmul(weights[0], X) + bias[0]
            A1 = tanh(Z1)
            Z2 = np.matmul(weights[1], A1) + bias[1]
            A2 = tanh(Z2)
            Z3 = np.matmul(weights[2], A2) + bias[2]
            A3 = tanh(Z3)

            cost = -(1. / lt) * (np.sum(np.multiply(np.log(A3), Y)) + np.sum(np.multiply(np.log(1 - A3), (1 - Y))))
            dZ3 = A3 - Y
            dW3 = (1. / m) * np.matmul(dZ3, A2.T)
            db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

            dA2 = np.matmul(weights[2].T, dZ3)
            dZ2 = dA2 * tanh(Z2) * (1 - tanh(Z2))
            dW2 = (1. / m) * np.matmul(dZ2, A1.T)
            db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.matmul(weights[1].T, dZ2)
            dZ1 = dA1 * tanh(Z1) * (1 - tanh(Z1))
            dW1 = (1. / m) * np.matmul(dZ1, X.T)
            db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

            weights[2] = weights[2] - lr * dW3
            bias[2] = bias[2] - lr * db3
            weights[1] = weights[1] - lr * dW2
            bias[1] = bias[1] - lr * db2
            weights[0] = weights[0] - lr * dW1
            bias[0] = bias[0] - lr * db1
            updated_weights = []
            updated_weights.append(weights[0])
            updated_weights.append(weights[1])
            updated_weights.append(weights[2])
            updated_bias = []
            updated_bias.append(bias[0])
            updated_bias.append(bias[1])
            updated_bias.append(bias[2])





    if activation == 'relu':
        for i in range(2000):
            Z1 = np.matmul(weights[0], X) + bias[0]
            A1 = relu(Z1)
            Z2 = np.matmul(weights[1], A1) + bias[1]
            A2 = relu(Z2)
            Z3 = np.matmul(weights[2], A2) + bias[2]
            A3 = relu(Z3)

            cost = -(1. / lt) * (np.sum(np.multiply(np.log(A3), Y)) + np.sum(np.multiply(np.log(1 - A3), (1 - Y))))
            dZ3 = A3 - Y
            dW3 = (1. / m) * np.matmul(dZ3, A2.T)
            db3 = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)

            dA2 = np.matmul(weights[2].T, dZ3)
            dZ2 = dA2 * relu(Z2) * (1 - relu(Z2))
            dW2 = (1. / m) * np.matmul(dZ2, A1.T)
            db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.matmul(weights[1].T, dZ2)
            dZ1 = dA1 * relu(Z1) * (1 - relu(Z1))
            dW1 = (1. / m) * np.matmul(dZ1, X.T)
            db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

            weights[2] = weights[2] - lr * dW3
            bias[2] = bias[2] - lr * db3
            weights[1] = weights[1] - lr * dW2
            bias[1] = bias[1] - lr * db2
            weights[0] = weights[0] - lr * dW1
            bias[0] = bias[0] - lr * db1
            updated_weights = []
            updated_weights.append(weights[0])
            updated_weights.append(weights[1])
            updated_weights.append(weights[2])
            updated_bias = []
            updated_bias.append(bias[0])
            updated_bias.append(bias[1])
            updated_bias.append(bias[2])

    return updated_weights, updated_bias

load_data("file:///C:/Users/Student/Downloads/mnist_train.csv")



# ***************************************************************problem 5************************************************************
import random
def forward(x, model):
    # Input to hidden
    h = x @ model['W1']
    # ReLU non-linearity
    h[h < 0] = 0

    # Hidden to output
    prob = sigmoid(h @ model['W2'])

    return h, prob

def backward(model, xs, hs, errs):


    dW2 = hs.T @ errs

    # Get gradient of hidden layer
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0

    dW1 = xs.T @ dh

    return dict(W1=dW1, W2=dW2)

def sgd(model, X_train, y_train, minibatch_size,n_iter):
    for iter in range(n_iter):
        print('Iteration {}'.format(iter))

        # Randomize data point
        X_train, y_train = shuffle(X_train, y_train)

        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current minibatch/chunk
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]

            model = sgd_step(model, X_train_mini, y_train_mini)

    return model

def sgd_step(model, X_train, y_train):
    grad = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()

    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        # Learning rate: 1e-4
        model[layer] += 1e-4 * grad[layer]

    return model

def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        # Create probability distribution of true label
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.

        # Compute the gradient of output layer
        err = y_true - y_pred

        # Accumulate the informations of minibatch
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs.append(h)
        errs.append(err)

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs), np.array(errs))

def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = shuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches



def momentum(model, X_train, y_train, minibatch_size,n_iter,alpha):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]

    return model