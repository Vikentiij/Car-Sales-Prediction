from optparse import Values
import PySimpleGUI as sg
import pandas as pd
import numpy as np
from scripttwo import Script
 

# Add some color
# to the window
sg.theme('DarkAmber')  
#Gender, Age, Annual Salary, Credit Card Debt, Net Worth  
layout = [
    [sg.Text("Please enter the credits to predict car sales price")],
    [sg.Text("Gender: 0 is female, 1 is male")],
    [sg.Input("")],
    [sg.Text("Age")],
    [sg.Input("")],
    [sg.Text("Annual Salary")],
    [sg.Input("")],
    [sg.Text(" Credit card debt")],
    [sg.Input("")],
    [sg.Text("Net worth ")],
    [sg.Input("")],  
    
    [sg.Text("Predict car price")],[sg.Text('', text_color='red',key='predictionKey')],     
    [sg.Text("Error percentage: ")],  [sg.Text('', text_color='red',key='errorKey')], 
    [sg.Button("Predict"),sg.Button("Cancel")]   
]
# layout2=[
#     [sg.Text("please enter No of Nuerons"),sg.Input("") ],    
#     [sg.Text("ErrorPercentage: ")],  [sg.Text('',font=('Helvetica', 18), text_color='red',key='error')],  
#     [sg.Button("ANN")]   
# ]


# Create the window
#window = sg.Window("Car Sales Prediction using ANN", layout)
form = sg.FlexForm('Predictor', default_button_element_size=(5, 2), auto_size_buttons=False, grab_anywhere=False)
form.Layout(layout)

 

# Create an event loop
while True:
    event, Values = form.Read() 
    # End program if user closes window or
    # presses the OK button
    if event == "Plot":             
        purchaseAmount=Script(50,Values)
        form.FindElement('predictionKey').Update(purchaseAmount[0])   
        form.FindElement('errorKey').Update(purchaseAmount[1]) 
              
        form.Read()        
        break   
     
    elif event == "Cancel" or event == sg.FORM_CLOSE:               
        break