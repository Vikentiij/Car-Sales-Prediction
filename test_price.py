from testing_model import NN
import numpy as np

# test cases changing the neurons in the hidden layer
BEST_CASE = 0.00018446765986626916 #NN(25)
#input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])

def test_first_layer():
    #NN(5) 
    #NN(15)
    # print(NN.MSE())
    assert NN(15) <= BEST_CASE


def test_second_layer():
    #NN(25) 
    # print(NN.MSE())
    assert NN(35) <= BEST_CASE
    #print()

def test_third_layer():
    #NN(50) 
    # print(NN.MSE())
    assert NN(50) <= BEST_CASE
