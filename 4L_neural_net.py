import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from Corneal_Data_Creation import x_train, x_test, y_train, y_test


#activation functions

def sigmoid(x):
        return 1/(1+np.exp(-x))

def relu(x):
        return np.maximum(0,x)

#Parameters

i_nodes = 2
h1_nodes = 2
h2_nodes = 2
o_nodes = 2

e = 0.01

epochs = 10000

#Initalize weights and biases 
w1 = np.random.randn(i_nodes, h1_nodes) * np.sqrt(2 / i_nodes)
w2 = np.random.randn(h1_nodes, h2_nodes) * np.sqrt(2 / h1_nodes)
w3 = np.random.randn(h2_nodes, o_nodes) * np.sqrt(2 / h2_nodes)

b1 = np.zeros((1, h1_nodes))
b2 = np.zeros((1, h2_nodes))
b3 = np.zeros((1, o_nodes)) 

#Training Loop

losses = []

for i in range(epochs):
        
        #forward pass
        h1 = relu(x_train.dot(w1) + b1)
        h2 = relu(h1.dot(w2) + b2)
        output = sigmoid(h2.dot(w3) + b3)
        
        #binary cross entropy loss
        loss = -np.mean(y_train * np.log(output + 1e-8) + (1 - y_train) * np.log(1 - output + 1e-8))
        losses.append(loss)
        
        
        #calculating new gradients 
        grad_pred = (output - y_train) / y_train.shape[0]
        
        grad_w3 = h2.T.dot(grad_pred)
        grad_b3 = np.sum (grad_pred, axis = 0, keepdims = True) 
        
        grad_h2 = grad_pred.dot(w3.T)
        grad_h2[h2<=0] *= 0.01
        grad_w2 = h1.T.dot(grad_h2)
        grad_b2 = np.sum(grad_h2, axis = 0, keepdims = True) 
        
        grad_h1 = grad_h2.dot(w2.T)
        grad_h1[h1<= 0 ] *= 0.01
        grad_w1 = x_train.T.dot(grad_h1)
        grad_b1 = np.sum(grad_h1, axis = 0, keepdims = True)         
        
        w1 = w1 - grad_w1 * e        
        w2 = w2 - grad_w2 * e
        w3 = w3 - grad_w3 * e       
        
        b1 = b1 - grad_b1 * e
        b2 = b2 - grad_b2 * e
        b3 = b3 - grad_b3 * e
        
        

#function to test neural network

def test (inputs, outputs):
        
        #forward pass
        h1 = relu(inputs.dot(w1) + b1)
        h2 = relu(h1.dot(w2) + b2)
        o_preds = sigmoid(h2.dot(w3) + b3)
        
        #binary cross entropy loss
        loss = -np.mean(outputs * np.log(o_preds + 1e-8) + (1 - outputs) * np.log(1 - o_preds + 1e-8)) 
        print(loss)
        
        #convert binary predictions to 0 or 1 
        pred_binary = (o_preds > 0.5).astype(int)
        
        #compute accuacy 
        correct = np.sum(np.all(pred_binary == outputs, axis=1))
        total = len(outputs)
        accuracy = np.mean(pred_binary == outputs) * 100
        print(accuracy) 

        

test(x_test, y_test)

import matplotlib.pyplot as plt

# Plot the loss over epochs
plt.figure(figsize=(8,5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()


