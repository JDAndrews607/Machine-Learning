#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:34:34 2024

@author: jeremy
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(data_location):
    '''
    This function will parse a csv file and
    set each row as a tensor to be returned
    '''
    tensors = []
    with open(data_location, 'r') as file:
        data = csv.reader(file, delimiter = ',')
        #next(data)      #skips column header
        for line in data:
            print(line)
            tensors.append(line)
    return tensors


'''
#############################################################
For single variable systems
#############################################################
'''

def SingleVar_GradiantDescent(x_train, y_train, m, b, learningRate = 1.0e-2):
    size = x_train.shape[0]
    dj_dm = 0   #partiel derivative of w
    dj_db = 0   #partial derivative of b
    predicted_y = 0
    
    for count in range(10000):
        for i in range(size):
            predicted_y = m * x_train[i] + b
            dj_dm_i = (predicted_y - y_train[i]) * x_train[i]
            dj_dm += dj_dm_i
            dj_db_i = (predicted_y - y_train[i])
            dj_db += dj_db_i
        dj_dm /= size
        dj_db /= size
            
        m = m - learningRate * dj_dm
        b = b - learningRate * dj_db    

    return m, b

def train_singleVarRegression(x_train, y_train, m = 0.01, b = 0.01, count = 0):
    
    count += 1
    print("Iteration ", count)
    cost = 0    

    numExamples = len(x_train)  # can also use x_train.shape[0]
    if count == 1:
        print("You have ", numExamples, " training examples")

    
    #To calculate the predicted dataset
    predicted_y = np.zeros(numExamples) #sets empty array to size of training set
    for i in range(numExamples):
        predicted_y[i] = m * x_train[i] + b
        
    #Checks predicted dataset against training set for accuracy
    for i in range(numExamples):
        cost += (predicted_y[i] - y_train[i])**2
    cost /= (2*numExamples)
    print(cost)

    if cost < 1:
        plt.scatter(x_train, y_train, label = "Training Data")
        plt.plot(x_train, predicted_y, label = "Model Data")
        plt.legend(shadow = True, fancybox = True)
        plt.title("Insert Title")
        plt.xlabel("x axis label")
        plt.ylabel("y axis label")
        plt.show()
        print(f'cost of {cost} found when m = {m} and b = {b}')
        print(f'Use equation: {m}x+{b}')
    else:
        m, b = SingleVar_GradiantDescent(x_train, y_train, m, b)
        print(m, b)
        train_singleVarRegression(x_train, y_train, m, b, count)
        



'''
#############################################################
For multivariable systems
#############################################################
'''

def multiVar_GradiantDescent(x_train, y_train, m_array, b, learningRate = 1.0e-2):
    dj_dm = np.zeros((x_train.shape[1],))
    dj_db = 0
    
    for count in range(10000):
        for i in range(x_train.shape[0]):
            error = (np.dot(x_train[i], m_array)+b) - y_train[i]
            for j in range(x_train.shape[1]):
                dj_dm[j] += error * x_train[i, j]
            dj_db += error
        dj_dm /= len(x_train)
        dj_db /= len(x_train)
    
    print(m_array)
    print(dj_dm)
    m_array -= learningRate * dj_dm
    b -= learningRate * dj_db
    
    return m_array, b

def train_multiVarRegression(x_train, y_train, m_array, b = 0.01, count = 0):
    
    count += 1
    print("Iteration ", count)
    cost = 0
    
    numExamples = x_train.shape[0]
    if count == 1:
        print(f'You have {numExamples} training examples')
    
    predicted_y = np.zeros(numExamples)
    for i in range(numExamples):
        print("x[i]: ", x_train[i], "m_array: " ,m_array)
        predicted_y[i] = np.dot(x_train[i], m_array) + b
        cost += (predicted_y[i] - y_train[i])**2
    cost /= (2*numExamples)

    if cost < 1:
        plt.scatter(x_train, y_train, label = "Training Data")
        plt.plot(x_train, predicted_y, label = "Model Data")
        plt.legend(shadow = True, fancybox = True)
        plt.title("Insert Title")
        plt.xlabel("x axis label")
        plt.ylabel("y axis label")
        plt.show()
        print(f'cost of {cost} found when b = {b}')
        print(f'Use equation: mx+{b}')
    else:
        b = multiVar_GradiantDescent(x_train, y_train, m_array, b)
        print(m_array, b)
        train_multiVarRegression(x_train, y_train, m_array, b, count)
