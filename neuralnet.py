import os
import pickle
import time

import cv2

from dataset.mnist import load_mnist
from trainer import Trainer
from useful_functions import *


class Neuralnet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, file_name='params.pkl'):
        self.file_name = file_name
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        if os.path.isfile(file_name):
            self.load_params()
        else:
            (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)

            ret, x_train = cv2.threshold(x_train, 0, 1, cv2.THRESH_BINARY)
            ret, x_test = cv2.threshold(x_test, 0, 1, cv2.THRESH_BINARY)

            trainer = Trainer(self, x_train, t_train, x_test, t_test,
                              epochs=200, batch_size=100, evaluate_sample_num_per_epoch=20)

            print('start training ...')
            time.sleep(2)
            trainer.train()

            self.save_params()

    def predict(self, x, verbose=False):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        if verbose:
            print('I think it\'s ... ' + str(np.argmax(y)) + ' (' + str(round(np.max(y), 5) * 100) + '%)')

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def save_params(self):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(self.file_name, 'wb') as f:
            pickle.dump(params, f)
        print('complete saving params')

    def load_params(self):
        with open(self.file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        print('complete loading params')
