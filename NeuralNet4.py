#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
from sklearn.preprocessing import LabelBinarizer
from impyute.imputation.cs import mice
import math

#sdf = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])

class NeuralNet:
    def __init__(self, raw_input, function, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input, function)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    "Modified to include tanh and ReLu"
    def __activation(self, x, function):
        if function == "sigmoid":
            return self.__sigmoid(x)
        elif function == "tanh":
            return self.__tanh(x)
        elif function == "ReLu":
            return self.__ReLu(x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    "Modified to include tanh and ReLu"
    def __activation_derivative(self, x, function):
        if function == "sigmoid":
            return self.__sigmoid_derivative(x)
        elif function == "tanh":
            return self.__tanh_derivative(x)
        elif function == "ReLu":
            return self.__ReLu_derivative(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    "Added tanh and ReLu activation functions"
    def __tanh(self, x):
        return np.tanh(x)

    def __ReLu(self, x):
        return np.maximum(0,x)

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return (1-np.square(np.tanh(x)))

    def __ReLu_derivative(self, x):
        return 1*(x>0)
    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X, function):
        #Handle null and missing values
        # print("X before preprocess \n"+str(X))
        fill_NaN = SimpleImputer(missing_values='?', strategy='most_frequent')
        imputed_X = pd.DataFrame(fill_NaN.fit_transform(X))

        encoded_X = imputed_X.apply(LabelEncoder().fit_transform)
        if function == "tanh":
            #tanh data normalized btwn -1 & 1
            proc_X = encoded_X.replace(0,-1)
        else:
            #sigmoid & ReLu data normalized btwn 0 & 1
            proc_X = encoded_X

        proc_X.columns = X.columns
        proc_X.index = X.index

        # print("X after preprocess \n"+str(proc_X))
        return proc_X

    # Below is the training function

    #sig=.001, tanh=.15, ReLu=.05
    def train(self, function, max_iterations = 1000, learning_rate = 0.1):
        for iteration in range(max_iterations):
            out = self.forward_pass(function)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, function)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        # print("TrainPred: " + str(out))
        if function == "sigmoid":
            print("***********************************SIGMOID***********************************")
        if function == "tanh":
            print("***********************************TANH************************************")
        if function == "ReLu":
            print("************************************RELU************************************")
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)/len(out)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self,function):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        # print("in1: " +str(self.X)+str(self.w01))
        self.X12 = self.__activation(in1,function)
        in2 = np.dot(self.X12, self.w12,)
        # print("in2: " +str(self.X12)+str(self.w12))
        self.X23 = self.__activation(in2,function)
        in3 = np.dot(self.X23, self.w23)
        # print("in3: " +str(self.X23)+str(self.w23))
        out = self.__activation(in3,function)
        # print("out: " + str(out))
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    "Modified to include tanh and ReLu"
    def compute_output_delta(self, out, function):
        delta_output = (self.y - out) * (self.__activation_derivative(out, function))
        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, function):
        delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__activation_derivative(self.X23,function))
        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, function):
        delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__activation_derivative(self.X12,function))
        self.delta12 = delta_hidden_layer1


    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, raw_input, function, header = True):
        test_dataset = self.preprocess(raw_input, function)
        ncols = len(test_dataset.columns)
        nrows = len(test_dataset.index)
        self.X = test_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = test_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        # print("test_dataset: " + str(test_dataset))

        pred = self.forward_pass(function)
        # print("Pred: " + str(pred))
        if function == "sigmoid":
            pred[pred>.5] = 1
            pred[pred<=.5] = 0
        if function == "tanh":
            pred[pred>0] = 1
            pred[pred<=0] = -1
        if function == "ReLu":
            pred[pred>0] = 1
            pred[pred<=0] = 0
        # print("P: " + str(pred))
        error = 0.5 * np.power((pred - self.y), 2)
        # print("out= " + str(self.y))


        fill_NaN = SimpleImputer(missing_values='?', strategy='most_frequent')
        export_csv = pd.DataFrame(fill_NaN.fit_transform(raw_input))
        strPred = ['republican' if i == 1 else 'democrat' for i in pred]
        export_csv['prediction'] = strPred
        if function == "sigmoid":
            export_csv.to_csv (r'C:\Users\domin\Desktop\NEWF19\Machine Learning\HW2\Assignment2\export_csv_sig', index = None, header=True)
        if function == "tanh":
            export_csv.to_csv (r'C:\Users\domin\Desktop\NEWF19\Machine Learning\HW2\Assignment2\export_csv_tanh', index = None, header=True)
        if function == "ReLu":
            export_csv.to_csv (r'C:\Users\domin\Desktop\NEWF19\Machine Learning\HW2\Assignment2\export_csv_ReLu', index = None, header=True)

        return error


if __name__ == "__main__":
    #neural_network = NeuralNet("train.csv")
    df = pd.read_csv("house_votes_84.csv")
    # print(df.describe())
    train, test = train_test_split(df, test_size=0.2)

    neural_network = NeuralNet(train, function ="sigmoid")
    neural_network.train(function ="sigmoid", learning_rate = 0.001)
    testError1 = neural_network.predict(test, function ="sigmoid")
    # print("Terror1= " + str(testError1))
    print("Test Error= " + str(np.sum(testError1)/len(testError1)))

    neural_network = NeuralNet(train, function ="tanh")
    neural_network.train(function ="tanh", learning_rate = 0.1)
    testError2 = neural_network.predict(test, function ="tanh")
    # print("Terror2= " + str(testError2))
    print("Test Error= " + str(np.sum(testError2)/len(testError2)))
    
    neural_network = NeuralNet(train, function ="ReLu")
    neural_network.train(function ="ReLu", learning_rate = 0.01)
    testError3 = neural_network.predict(test, function ="ReLu")    
    # print("Terror3= " + str(testError3))
    print("Test Error= " + str(np.sum(testError3)/len(testError3)))