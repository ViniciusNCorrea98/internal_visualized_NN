import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle
import copy


train_dict={}

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons,
        weight_regularizer_l1 = 0, weight_regularizer_l2 = 0,
        bias_regularizer_l1 = 0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2

        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2


    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):


        """To calculate the gradients' values!"""
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights<0] = -1
            self.dweights += self.weight_regularizer_l1*dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2*self.weight_regularizer_l2*self.weights

        if self.bias_regularizer_l1> 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1*dL1
        if self.bias_regularizer_l2>0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases



class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):


        self.inputs = inputs

        if not training:
            self.output = inputs.copy()

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape)/self.rate

        self.output = inputs * self.binary_mask

    def backward(self,dvalues):
        self.dinputs = dvalues * self.binary_mask


class Layer_Input:

    def forward(self, inputs):
        self.output = inputs

class Activation_ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximun(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copu()
        self.dinputs[self.inputs <= 0] = 0


    def predictions(self, outputs):
        return outputs

class Activation_Softmax:

    def forward(self, inputs, training):

        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = prob

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)
            jacobian_mtx = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[i] = np.dot(jacobian_mtx, single_dvalues)



    def prediction(self, outputs):
        return np.argmax(outputs, axis=1)


