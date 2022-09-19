from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.python.keras
import os
import graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ann_visualizer.visualize import ann_viz
from graphviz import Digraph
from keras import Sequential
from keras.layers import Dense

 
def Script(N,dynamicInput):
# data import
    data = pd.read_csv('car_sales_dataset.txt', encoding='ISO-8859-1')
    print(data)
    os.environ["PATH"] += os.pathsep + \
        'C:/Users/vijay/Downloads/windows_10_msbuild_Release_graphviz-5.0.1-win32/Graphviz/bin'
    # Plot data
    sns.pairplot(data)  # first graph
  #  plt.show(block=True)

    # Create input dataset from data
    inputs = data.drop(['Customer_Name', 'Customer_Email',
                        'Country', 'Purchase_Amount'], axis=1)
    New_inputs=data['Purchase_Amount']
    # Show Input Data
 #   print('raw filtered input', inputs)
    # Show Input Shape
  #  print("Input data Shape=", inputs.shape)

    # Create output dataset from data
    output = data['Purchase_Amount']
    # Show Output Data
   # print(output)
    # Transform Output
    output = output.values.reshape(-1, 1)
    # Show Output Transformed Shape
    #print("Output Data Shape=", output.shape)

    # Scale input
    scaler_in = MinMaxScaler()
    input_scaled = scaler_in.fit_transform(inputs)
    #print('transformed Input layer')
    #print(input_scaled)

    # Scale output
    scaler_out = MinMaxScaler()
    output_scaled = scaler_out.fit_transform(output) 
    #print('transformed output layer')
   # print(output_scaled)
    # Create model
    model = Sequential()
    model.add(Dense(N, input_dim=5, activation='relu'))
    model.add(Dense(N, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # Train model
    model.compile(optimizer='adam', loss='mean_squared_error')
    epochs_hist = model.fit(input_scaled, output_scaled,  epochs=20,
                            batch_size=10,    verbose=1, validation_split=0.2)
   # print(epochs_hist.history.keys())

    # Plot the training graph to see how quickly the model learns
    # plt.plot(epochs_hist.history['loss'])
    # plt.plot(epochs_hist.history['val_loss'])
    # plt.title('Model Loss Progression During Training/Validation')
    # plt.ylabel('Training and Validation Losses')
    # plt.xlabel('Epoch Number')
    # plt.legend(['Training Loss', 'Validation Loss'])
    # plt.show(block=True)

    ann_viz(model, view=True, filename='network.gv',
            title='Model Loss Progression During Training/Validation')
    def Prediction(dynamicInput):
        print(dynamicInput)
        input_test_sample = np.array([list(dynamicInput.values())])
        #input_test_sample2 = np.array([[1, 46.73, 61370.67, 9391.34, 462946.49]])
        # Scale input test sample data
        input_test_sample_scaled = scaler_in.transform(input_test_sample)

        # Predict output
        output_predict_sample_scaled = model.predict(input_test_sample_scaled)
    # Print predicted output
        print('Predicted Output (Scaled) =', output_predict_sample_scaled)
        # Unscale output
        p = scaler_out.inverse_transform(output_predict_sample_scaled)
        print('Predicted Output / Purchase Amount ', p)
        return p
   
    def ANN(N):
        N=10
        model.add(Dense(N, input_dim=5, activation='relu'))
        model.add(Dense(N, activation='relu'))
        model.add(Dense(1, activation='linear'))
        print("model output")
        print(model.summary())  

        x_train, x_test, y_train, y_test = train_test_split(
            inputs, New_inputs, test_size=0.2)
        # call for prediction function to find purchase amount for
        # p = Prediction(x_test)
        # error_percentage = mean_squared_error(y_test, p)
           #Defining MAPE function
        def MAPE(Y_actual,Y_Predicted):  
            mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
            return mape
        linear_model = LinearRegression().fit(x_train , y_train)
    
                        #Predictions on Testing data
        LR_Test_predict = linear_model.predict(x_test) 
    
                        # Using MAPE error metrics to check for the error rate and accuracy level
        error_percentage= MAPE(y_test,LR_Test_predict)
        return error_percentage
        print("MAPE: ",error_percentage)
    if(dynamicInput!=None)&(N!=None):  
             
        Prediction=Prediction(dynamicInput)
        ErrorPercentage=ANN(N)
        print('result: ',Prediction,ErrorPercentage)
        result=[Prediction, ErrorPercentage]
     
    return result

