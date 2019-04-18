import numpy as np
import pickle
import seaborn as sns

epsilon = 1e-8
beta1 = 0.9
beta2 = 0.999

def ReLU(z):
    return z * (z > 0)

def ReLU_prime(z):
    return 1. * (z > 0)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
     return sigmoid(z)*(1-sigmoid(z))

def linear(z):
    return z

def linear_prime(z):
    return z >= 0

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_prime(z):
    return 1 - (tanh(z))**2

def softplus(z):
    return np.log(1 + np.exp(z))

def softplus_prime(z):
    return sigmoid(z)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=0, keepdims=True)

def cost_binary_cross_entropy(Y_hat, Y, weights=None):
    if weights is None:
        weights = np.ones((2,1))
    m = Y.shape[1]
    #print(str(weights[0]), str(weights[1]))
    return - 1/m * np.sum(Y*np.log(Y_hat + epsilon) * weights[0] + (1 - Y) * np.log(1 - Y_hat + epsilon) * weights[1])

def cost_binary_cross_entropy_derivative(AL, Y, weights=None):
    if weights is None:
        weights = np.ones((2,1))
    #print(str(weights[0]), str(weights[1]))
    return -Y/(AL + epsilon) * weights[0] + (1-Y)/(1 - AL + epsilon) * weights[1]

def non_binary_cross_entropy(Y_hat, Y):
    return - 1/Y_hat.shape[1] * np.sum(Y * np.log(Y_hat + epsilon))

def non_binary_cross_entropy_derivative(AL, Y):
    return AL - Y

def cost_mse(Y_hat, Y):
    m = Y.shape[1]
    return 1/(2*m) * np.sum(Y_hat - Y)**2

def cost_mse_derivatie(AL, Y):
    return AL - Y

# return activation function by given name
def get_activation_function_by_name(name):
    if name == 'sigmoid':
        return sigmoid
    if name == 'ReLU':
        return ReLU
    if name == 'linear':
        return linear
    if name == 'tanh':
        return tanh
    if name == 'softplus':
        return softplus
    if name == 'softmax':
        return softmax

def get_activation_derivative_function_by_name(name):
    if name == 'sigmoid':
        return sigmoid_prime
    if name == 'ReLU':
        return ReLU_prime
    if name == 'linear':
        return linear_prime
    if name == 'tanh':
        return tanh_prime
    if name == 'softplus':
        return softplus_prime

def cost_function_by_name(name, binary=False):
    if name == 'mse':
        return cost_mse
    if name == 'cross-entropy' and binary:
        return cost_binary_cross_entropy
    if name == 'cross-entropy' and not binary:
        return non_binary_cross_entropy

def cost_function_derivative_by_name(name, binary=False):
    if name == 'mse':
        return cost_mse_derivatie
    if name == 'cross-entropy' and binary:
        return cost_binary_cross_entropy_derivative
    if name == 'cross-entropy' and not binary:
        return non_binary_cross_entropy_derivative

def sgn(x):
    return x and (1, -1)[x < 0]

class NeuralNetwork:
    # input_units_count - size of one input vector (input_units_count, 1)
    def __init__(self, input_units_count):

        # the number of neurons in each layer
        self.layers_structure = []
        self.layers_structure.append(input_units_count)

        # next added layer will have l = 1
        self.current_number = 1

        # dictionary of activation functions
        self.activation_functions = {}
        self.activation_function_derivatives = {}

        # dictionary of weights and biases
        self.parameters = {}
        # dictionary of betas and gammas
        self.batch_norm_parameters = {}
        self.batch_norm_caches = {}
        # dictionary for mu and sigma, calculated by mini-batches
        self.batch_biases = {}

        # cost function initialize with applying model
        self.cost_function = None
        self.cost_function_derivative = None

        self.isSoftmax = False

        # Regularization
        self.L2 = 0
        self.L1 = 0

        np.random.seed(4)

    def add(self, count_of_units, name_of_activation, batch_normalization=False):
        self.layers_structure.append(count_of_units)

        # adds function to dictionary
        self.activation_functions['g' + str(self.current_number)] = get_activation_function_by_name(name_of_activation)
        self.activation_function_derivatives['dg' + str(self.current_number)] = \
            get_activation_derivative_function_by_name(name_of_activation)

        gamma = None
        beta = None
        # initialize parameters for batch normalization
        if batch_normalization:
            gamma = np.random.randn(count_of_units, 1)
            beta = np.zeros((count_of_units, 1))

        self.batch_norm_parameters['gamma' + str(self.current_number)] = gamma
        self.batch_norm_parameters['beta' + str(self.current_number)] = beta

        self.current_number += 1

    # user should call this method after adding layers. Weights and biases will be generated here
    def apply_model(self, cost_function_name):
        # L - count of hidden layers + output layer
        L = len(self.activation_functions)

        for i in range(1, L + 1):
            self.parameters['W' + str(i)] = np.random.randn(self.layers_structure[i], self.layers_structure[i-1]) #* 0.01
            self.parameters['b' + str(i)] = np.zeros((self.layers_structure[i], 1))

        self.cost_function = cost_function_by_name(cost_function_name, binary=self.activation_functions['g' + str(L)] == sigmoid)
        self.cost_function_derivative = cost_function_derivative_by_name(cost_function_name,
                                                                         binary=self.activation_functions['g' + str(L)] == sigmoid)

        if self.activation_functions['g' + str(L)] == softmax:
            self.isSoftmax = True

    # needed for cross-validation. reset all weights, biases, gammas and betas
    def __reset_model_parameters__(self):
        # L - count of hidden layers + output layer
        L = len(self.activation_functions)

        for i in range(1, L + 1):
            self.parameters['W' + str(i)] = np.random.randn(self.layers_structure[i], self.layers_structure[i-1]) #* 0.01
            self.parameters['b' + str(i)] = np.zeros((self.layers_structure[i], 1))

            if self.batch_norm_parameters['gamma' + str(i)] is not None:
                self.batch_norm_parameters['gamma' + str(i)] = np.random.randn(self.layers_structure[i], 1)
                self.batch_norm_parameters['beta' + str(i)] = np.zeros((self.layers_structure[i], 1))

    # compute output of NN for given set X
    # if train_prop=False => mu and sigma^2 should be used by mini-batches
    def __forward_prop__(self, X, t, train_prop=True):
        L = len(self.parameters) // 2
        A = X
        zs = []
        activations = [A]
        for i in range(1, L+1):
            A_prev = A
            Wi = self.parameters['W' + str(i)]
            bi = self.parameters['b' + str(i)]

            Z = np.dot(Wi, A_prev) + bi

            # check for batch_normalization
            if self.batch_norm_parameters['gamma' + str(i)] is not None:
                # if first mini-batch
                # if t == 1:
                #     self.batch_biases['mu' + str(i)] = np.zeros(bi.shape)
                #     self.batch_biases['sigma' + str(i)] = np.zeros(bi.shape)

                Z = self.__batch_norm_forward__(Z, i, train_prop, t)

            A = self.activation_functions['g' + str(i)](Z)

            zs.append(Z)
            activations.append(A)

        return activations[-1], zs, activations

    # batch-normalization helper for forward propagation
    # z - vector to normalize, l - number of layer
    def __batch_norm_forward__(self, z, l, train_prop, t):
        gamma = self.batch_norm_parameters['gamma' + str(l)]
        beta = self.batch_norm_parameters['beta' + str(l)]

        if train_prop:
            mu = 1 / z.shape[1] * np.sum(z, axis=1, keepdims=True)

            imean = (z - mu)

            sigma = 1 / z.shape[1] * np.sum(imean ** 2, axis=1, keepdims=True)

            ivar = 1 / (np.sqrt(sigma + epsilon))

            z_hat = imean * ivar

            self.batch_biases['mu' + str(l)] = mu
            self.batch_biases['ivar' + str(l)] = ivar
        else:
            mu = self.batch_biases['mu' + str(l)]
            ivar = self.batch_biases['ivar' + str(l)]
            imean = None

            z_hat = (z-mu) * ivar

        y = gamma * z_hat + beta

        cache = (imean, ivar, gamma, beta, z_hat)

        self.batch_norm_caches[str(l)] = cache

        return y

    # calculates cost function
    def __calculate_cost_function__(self, Y_hat, Y, weights):
        return self.cost_function(Y_hat, Y, weights)

    # implementation of back_propagation
    # returns dictionary with dW[1], db[1],...,dW[L], db[L]
    def __back_prop__(self, dAL, zs, activations):

        L = len(activations)
        dAL_copy = dAL
        m = dAL.shape[1]

        # result dictionary with gradients
        grads = {}
        for i in range(1, L):
            if self.activation_functions['g' + str(L - i)] != softmax:
                dZ = dAL_copy * self.activation_function_derivatives['dg' + str(L - i)](zs[- i])
            else:
                dZ = dAL_copy

            if self.batch_norm_parameters['gamma' + str(L-i)] is not None:
                dZ, dgamma, dbeta = self.__batch_norm_backward__(dZ, L-i)
                grads['dgamma' + str(L - i)] = dgamma
                grads['dbeta' + str(L - i)] = dbeta

            dWL = 1 / m * np.dot(dZ, activations[-i - 1].T)
            grads['dW' + str(L - i)] = dWL
            grads['db' + str(L - i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dAL_copy = np.dot(self.parameters['W' + str(L - i)].T, dZ)

            # Regularization adding to dW
            if(self.L2 > 0):
                #grads['dW' + str(L - i)] += (self.L2 / m * self.parameters['W' + str(L - i)])
                grads['dW' + str(L - i)] += (self.L2 * self.parameters['W' + str(L - i)])

            if(self.L1 > 0):
                for k in range(0, self.parameters['W' + str(L - i)].shape[0]):
                    for j in range(0, self.parameters['W' + str(L - i)].shape[1]):
                        #grads['dW' + str(L - i)] += self.L1/m * sgn(self.parameters['W' + str(L - i)][k][j])
                        grads['dW' + str(L - i)] += self.L1 * sgn(self.parameters['W' + str(L - i)][k][j])

        return grads

    # calculating dZ, dBeta, dGamma after batch_normalization
    def __batch_norm_backward__(self, dZ, l):
        m = dZ.shape[1]

        imean, ivar, gamma, beta, z_hat = self.batch_norm_caches[str(l)]

        dx_hat = dZ * gamma

        dsigma = np.sum(dx_hat * imean * (-0.5) * ivar ** 3, axis=1, keepdims=True)

        dmu = np.sum(dx_hat * (-1)*ivar) + dsigma * (-2)/m * np.sum(imean, axis=1, keepdims=True)

        dZ_result = dx_hat*ivar + dsigma * 2/m * imean + dmu/m

        dgamma = np.sum(dZ * z_hat, axis=1, keepdims=True)
        dbeta = np.sum(dZ, axis=1, keepdims=True)

        return dZ_result, dgamma, dbeta

    # update parameters (weights an biases) with given gradients and learning rate
    def __update_parameters__(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for i in range(1, L + 1):
            self.parameters['W' + str(i)] = self.parameters['W' + str(i)] - learning_rate * grads['dW' + str(i)]
            self.parameters['b' + str(i)] = self.parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]

            if self.batch_norm_parameters['gamma' + str(i)] is not None:
                self.batch_norm_parameters['gamma' + str(i)] =  \
                    (self.batch_norm_parameters['gamma' + str(i)] - learning_rate * grads['dgamma' + str(i)])
                self.batch_norm_parameters['beta' + str(i)] = \
                    (self.batch_norm_parameters['beta' + str(i)] - learning_rate * grads['dbeta' + str(i)])

    def get_regularization_adding_to_cost(self, m):
        sum_of_reg = 0
        # L2
        if(self.L2 > 0):
            for i in range(1, len(self.parameters) // 2 + 1):
                W = self.parameters['W' + str(i)]
                sum_of_reg += (self.L2/(2*m) * np.sum(np.dot(W.T, W)))

        # L1
        if(self.L1 > 0):
            for i in range(1, len(self.parameters) // 2 + 1):
                W = self.parameters['W' + str(i)]

                sum_of_reg += (self.L1/m * np.sum(np.abs(W)))

        return sum_of_reg

    # Initialize of V and S parameters for adam optimizer
    def __init_adam_parameters__(self):
        L = len(self.parameters) // 2

        # Vdw and Vdb parameters dictionary for all layer
        v = {}
        # the same with Sdw and Sb
        s = {}

        for i in range(1, L + 1):
            v['dW' + str(i)] = np.zeros(self.parameters['W' + str(i)].shape)
            v['db' + str(i)] = np.zeros(self.parameters['b' + str(i)].shape)

            s['dW' + str(i)] = np.zeros(self.parameters['W' + str(i)].shape)
            s['db' + str(i)] = np.zeros(self.parameters['b' + str(i)].shape)

            # initializing batch_norm parameters for all neuron in current layer if needed
            if self.batch_norm_parameters['gamma' + str(i)] is not None:
                v['dgamma' + str(i)] = np.zeros(self.parameters['b' + str(i)].shape)
                v['dbeta' + str(i)] = np.zeros(self.parameters['b' + str(i)].shape)

                s['dgamma' + str(i)] = np.zeros(self.parameters['b' + str(i)].shape)
                s['dbeta' + str(i)] = np.zeros(self.parameters['b' + str(i)].shape)

        # returns dictionaries with initialized by zeros parameters
        return v, s

    # train model with
    # X_train - train features, Y_train - answers for train set
    # L1 - L1 regularization, L2 - L2 regularization
    # X_test, Y_test - test set for monitoring overfitting
    def train(self, X_train, Y_train, learning_rate, epochs, L1=0, L2=0, X_test=None, Y_test=None,
              output=True, mini_batch_size=None, optimizer='sgd', lr_decay=1, weights=None):
        count_epochs = []
        train_costs = []
        test_costs = []
        # only for non-binary classifictaion
        if (Y_test is not None) and (self.isSoftmax):
            Y_test_args = np.argmax(Y_test, axis=0)
        else:
            Y_test_args = Y_test

        v = None
        s = None

        if optimizer == 'adam':
            v, s = self.__init_adam_parameters__()

        # Regularization
        self.L2 = L2
        self.L1 = L1

        max_right_answ = 0
        t = 0

        # if batch-size is not defined, then all train data is mini-batch
        if mini_batch_size is None:
            mini_batch_size = X_train.shape[1]

        t = 0

        # reducing of learning rate
        current_learning_coef = lr_decay
        corrected_lr = learning_rate
        for i in range(1, epochs + 1):
            t += 1
            corrected_lr *= current_learning_coef
            # preparing mini-batch
            #stacked_data = np.row_stack((X_train, Y_train)).T
            #np.random.shuffle(stacked_data)
           # mini_batches = [
            #    stacked_data[k: k + mini_batch_size]
            #    for k in range(0, X_train.shape[1], mini_batch_size)]
            # number of current mini-batch

            #for current_mini_batch in mini_batches:
                #current_mini_batch = current_mini_batch.T

                #X_batch = np.array(current_mini_batch[0: X_train.shape[0]])
                #Y_batch = np.array(current_mini_batch[X_train.shape[0]: X_train.shape[0]+Y_train.shape[0]])

                # AL - predictions on train set
                # zs, activations - cache for easier computing of backprop
            AL, zs, activations = self.__forward_prop__(X_train, t)

            dAL = self.cost_function_derivative(AL, Y_train, weights)

            # select suitable optimizer to update weights and biases
            if optimizer == 'sgd':
                self.__sgd_optimize__(dAL=dAL, zs=zs, activations=activations, learning_rate=corrected_lr)
            elif optimizer == 'adam':
                v, s = self.__adam_optimize__(dAL=dAL, zs=zs,activations=activations,learning_rate=corrected_lr,
                                   t=t, v=v, s=s)

            AL, zs, activations = self.__forward_prop__(X_train, -1, False)
            cost_train = self.__calculate_cost_function__(AL, Y_train, weights)

            # Adding regularization to cost function (also should be added to gradient)
            regularization = self.get_regularization_adding_to_cost(AL.shape[1])
            #cost_train += regularization

            count_epochs.append(i)
            train_costs.append(cost_train)

            if output:
                if X_test is not None:
                    AL_test, zs_test, act_test = self.__forward_prop__(X_test, -1, False)
                    cost_test = self.__calculate_cost_function__(AL_test, Y_test, weights)
                    #cost_test += regularization

                    predicted = self.predict(X_test)
                    right_answers = 0
                    if not self.isSoftmax:
                        for j in range(0, predicted.shape[1]):
                            right_answers += predicted[0][j] == Y_test_args[0][j]
                    else:
                        for j in range(0, predicted.shape[0]):
                            right_answers += predicted[j] == Y_test_args[j]

                    if right_answers > max_right_answ:
                        max_right_answ = right_answers

                    test_costs.append(cost_test)
                    if self.isSoftmax:
                        predicted_all_count = predicted.shape[0]
                    else:
                        predicted_all_count = predicted.shape[1]
                    print("Epoch %i: Train cost: %f  Test cost: %f, right answers: %i/%i"
                          %(i, cost_train, cost_test, right_answers, predicted_all_count))
                else:
                    print("Epoch %i: Train cost: %f " % (i, cost_train))

        return count_epochs, train_costs, test_costs, max_right_answ

    # SGD optimizer
    def __sgd_optimize__(self, dAL, zs, activations, learning_rate):
        grads = self.__back_prop__(dAL=dAL, zs=zs, activations=activations)
        self.__update_parameters__(grads=grads, learning_rate=learning_rate)

    # adam optimizer returns new values of v and s
    def __adam_optimize__(self, dAL, zs, activations, learning_rate, t, v, s):
        grads = self.__back_prop__(dAL=dAL, zs=zs, activations=activations)

        L = len(self.parameters) // 2

        # bias corrected dictionary of v and s
        v_cr = {}
        s_cr = {}

        for i in range(1, L + 1):
            # calculating of exponentially weight averages for V
            v['dW' + str(i)] = beta1 * v['dW' + str(i)] + (1 - beta1) * grads['dW' + str(i)]
            v['db' + str(i)] = beta1 * v['db' + str(i)] + (1 - beta1) * grads['db' + str(i)]
            # The same with S
            s['dW' + str(i)] = beta2 * s['dW' + str(i)] + (1 - beta1) * (grads['dW' + str(i)]**2)
            s['db' + str(i)] = beta2 * s['db' + str(i)] + (1 - beta1) * (grads['db' + str(i)]**2)

            if self.batch_norm_parameters['gamma' + str(i)] is not None:
                v['dgamma' + str(i)] = beta1 * v['dgamma' + str(i)] + (1 - beta1) * grads['dgamma' + str(i)]
                v['dbeta' + str(i)] = beta1 * v['dbeta' + str(i)] + (1 - beta1) * grads['dbeta' + str(i)]

                s['dgamma' + str(i)] = beta2 * s['dgamma' + str(i)] + (1 - beta2) * (grads['dgamma' + str(i)] ** 2)
                s['dbeta' + str(i)] = beta2 * s['dbeta' + str(i)] + (1 - beta2) * (grads['dbeta' + str(i)] ** 2)

            # calculation of corrected values
            v_cr['dW' + str(i)] = v['dW' + str(i)] / (1 - beta1 ** t)
            v_cr['db' + str(i)] = v['db' + str(i)] / (1 - beta1 ** t)

            s_cr['dW' + str(i)] = s['dW' + str(i)] / (1 - beta2 ** t)
            s_cr['db' + str(i)] = s['db' + str(i)] / (1 - beta2 ** t)

            if self.batch_norm_parameters['gamma' + str(i)] is not None:
                v_cr['dgamma' + str(i)] = v['dgamma' + str(i)] / (1 - beta1 ** t)
                v_cr['dbeta' + str(i)] = v['dbeta' + str(i)] / (1 - beta1 ** t)

                s_cr['dgamma' + str(i)] = s['dgamma' + str(i)] / (1 - beta2 ** t)
                s_cr['dbeta' + str(i)] = s['dbeta' + str(i)] / (1 - beta2 ** t)

            # updating of weights
            self.parameters['W' + str(i)] = self.parameters['W' + str(i)] - learning_rate * \
                                            (v_cr['dW' + str(i)] / (np.sqrt(s_cr['dW' + str(i)]) + epsilon))
            self.parameters['b' + str(i)] = self.parameters['b' + str(i)] - learning_rate * \
                                            (v_cr['db' + str(i)] / (np.sqrt(s_cr['db' + str(i)]) + epsilon))

            if self.batch_norm_parameters['gamma' + str(i)] is not None:
                self.batch_norm_parameters['gamma' + str(i)] = self.batch_norm_parameters['gamma' + str(i)] - \
                    learning_rate * (v_cr['dgamma' + str(i)] / (np.sqrt(s_cr['dgamma' + str(i)]) + epsilon))
                self.batch_norm_parameters['beta' + str(i)] = self.batch_norm_parameters['beta' + str(i)] - \
                    learning_rate * (v_cr['dbeta' + str(i)] / (np.sqrt(s_cr['dbeta' + str(i)]) + epsilon))

        return v, s

    # make prediction for given X
    def predict(self, X):
        AL, zs, act = self.__forward_prop__(X, t=-1, train_prop=False)
        L = len(self.parameters) // 2
        # if net has sigmoid activation at Lth layer => binary classification
        if self.activation_functions['g' + str(L)] == sigmoid:
            return AL >= 0.5
        # regression
        elif self.activation_functions['g' + str(L)] == ReLU or self.activation_functions['g' + str(L)] == linear:
            return AL
        else:
            return np.argmax(AL, axis=0)

    # save current model to the path /saved models/<file_name>
    def save_to_file(self, file_name):
        serialized_model = pickle.dumps(self)
        file = open('saved models/' + file_name, 'wb')
        file.write(serialized_model)
        file.close()
        # loaded_nn = pickle.loads(c)

    # reads model from file, deserialize it and returns object
    @staticmethod
    def read_model_from_file(file_name):
        file = open('saved models/' + file_name, 'rb')
        serialized_model = file.read()
        file.close()
        deserialized_model = pickle.loads(serialized_model)
        return deserialized_model

    # show heatmap with weight matrix for given layer
    def draw_heatmap(self, layer_to_vizualize=1):
        sns.heatmap(self.parameters['W' + str(layer_to_vizualize)])

    def cross_validation(self, X, Y, learning_rate, epochs, L1=0, L2=0,
              output=True, mini_batch_size=None, optimizer='sgd', count_range=5, weights=None):
        test_begin = 0
        test_end = X.shape[1] // count_range

        test_cost_avg = 0
        for i in range(count_range):
            if test_end > X.shape[1]:
                test_end = X.shape[1]

            self.__reset_model_parameters__()
            # copy X and Y (test range will drop)
            X_train = np.copy(X)
            Y_train = np.copy(Y)

            test_range = np.arange(test_begin, test_end)
            X_test = np.take(X, test_range, axis=1)
            Y_test = np.take(Y, test_range, axis=1)

            print("sum in test set: " + str(np.sum(Y_test)))

            X_train = np.delete(X_train, test_range, axis=1)
            Y_train = np.delete(Y_train, test_range, axis=1)

            epoch, train_cost, test_cost, max_right = \
                self.train(X_train, Y_train, learning_rate, epochs, L1, L2, X_test,
                           Y_test, output, mini_batch_size, optimizer,weights=weights)

            test_cost_avg += test_cost[-1]

            test_begin = test_end
            test_end += len(test_range)

        return test_cost_avg/count_range

    def get_parameters(self):
        return self.parameters

