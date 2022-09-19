import PySimpleGUI as sg
import numpy as np
from testing_model import Prediction


sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text("Please enter the credits to predict car sales price")],
        [sg.Text('Gender: 0 is female, 1 is male')],
        [sg.Input("")],
        [sg.Text('Age')],
        [sg.Input(" ")],
        [sg.Text('Annual Salary')],
        [sg.Input("")],
        [sg.Text('Credit card debt')],
        [sg.Input("")],
        [sg.Text('Net worth')],
        [sg.Input("")],

        [sg.Button('Predict'), sg.Button('Cancel')],
        [sg.Output(size=(10, 10), key='-OUTPUT-')] ]
        #[sg.Text(size=(10, 10), key='_TEXT_')] ]

# Create the Window
window = sg.Window('Car Sales Prediction', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    
    elif  event== "Predict": #script.InputLisI=value
        purshaseAmount=Prediction(values, 50)
        # window.FindElement('predictionKey').Update(purchaseAmount[0])
        # window.FindElement('errorKey').Update(purchaseAmount[1])
        window.Read()
        #window['_TEXT_'].Update(values[purshaseAmount])
# Object.values(object1)
#const object1 = {
 # a: 'somestring',
 # b: 42,
 #c: false
##};

#console.log(Object.values(object1));
#// expected output: Array ["somestring", 42, false]
##

        window.findElement('-OUTPUT-').Update(purshaseAmount[0])
        print('Predict sales is ', purshaseAmount)

window.close()

