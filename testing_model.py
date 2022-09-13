import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense


def NN(neuron_numbers):
    #Import data
    data = pd.read_csv('car_sales_dataset.txt', encoding='ISO-8859-1')


    #Create input dataset from data
    inputs = data.drop(['Customer_Name', 'Customer_Email', 'Country', 'Purchase_Amount'], axis = 1)
    
    #Create output dataset from data
    output = data['Purchase_Amount']
    #Transform Output
    output = output.values.reshape(-1,1)


    #Scale input
    scaler_in = MinMaxScaler()
    input_scaled = scaler_in.fit_transform(inputs)

    #Scale output
    scaler_out = MinMaxScaler()
    output_scaled = scaler_out.fit_transform(output)

    input_train, input_test, output_train, output_test= train_test_split(input_scaled, output_scaled, test_size=0.2)

    neuron_numbers=25
    #Create model
    model = Sequential()
    model.add(Dense(neuron_numbers, input_dim=5, activation='relu'))
    model.add(Dense(neuron_numbers, activation='relu'))
    model.add(Dense(1, activation='linear'))
    #print(model.summary())

    #Train model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
    epochs_hist = model.fit(input_train, output_train, epochs=20, batch_size=10, verbose=1, validation_data=(input_test, output_test))

    predictions = model.predict(input_test)
    MSE = mean_squared_error(output_test, predictions)
    print(f'Error: {MSE:.20f} %')

    return MSE

NN(25)








# Evaluate model
# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 
# ***(Note that input data must be normalized)***

#input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])
#input_test_sample2 = np.array([[1, 46.73, 61370.67, 9391.34, 462946.49]])

#Scale input test sample data
#input_test_sample_scaled = scaler_in.transform(input_test_sample)

#Predict output
#output_predict_sample_scaled = model.predict(input_test_sample_scaled)

#Print predicted output
#print('Predicted Output (Scaled) =', output_predict_sample_scaled)

#Unscale output
#output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
#print('Predicted Output / Purchase Amount ', output_predict_sample)


